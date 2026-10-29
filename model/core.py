"""Issue-stall model for the vector side: a register scoreboard plus FIFO units.

A bundle issues at the first cycle where every vector register it reads is ready and
every unit it needs can accept work (in-order VLIW with interlocks, HARDWARE_FACTS F9).

FIFO units (MXU, XLU, EUP) take work ops and return results through `vpop.*`:
- work op on unit u starts when u is free, occupies it `occupancy` cycles, and queues
  `results` entries that become poppable `latency` (+ j * `interval`) cycles after start;
- a pop waits for its unit's oldest entry; the popped vreg is ready `pop_latency` later. With
  `nonblocking_pop` the pop issues at once and only readers of its vreg wait for the entry.
MXU weights: `vmatpush.<msr>` streams into staging register msr (one push path per MXU,
`push_cycles` each, `push_cycles_f32` for f32 weights, which take two pushes per bf16 one,
`push_cycles_xpose` for transposed bf16 weights); a
`vmatmul.vlgmr.<msr>` first waits until those pushes finish plus `load_latency`.
"""

import re
from dataclasses import dataclass, field

VREG = re.compile(r"%\w+?_(vm\d+|v\d+)\b")
SREG = re.compile(r"%\w+?_(s\d+)\b")


@dataclass
class CoreParams:
    latency: dict = field(default_factory=dict)  # mnemonic prefix -> result latency (cycles)
    default_latency: int = 1
    units: dict = field(default_factory=dict)  # unit class -> params, see classify()
    issue_lead: float = 0.0  # cycles the front end may run ahead of queued vector ops (Scoreboard)


def _longest_prefix(mnemonic, table, default):
    best, best_len = default, -1
    for prefix, val in table.items():
        if mnemonic.startswith(prefix) and len(prefix) > best_len:
            best, best_len = val, len(prefix)
    return best


def classify(mnemonic):
    """(unit class, unit id, role) for FIFO-unit ops, else None.

    role: 'work' (produces results), 'pop', 'push' (MXU weight staging), 'xpose' (XLU transpose
    input: start/cont/end, one vreg each, two for `.binary`; results queue at `.end`)."""
    if mnemonic.startswith("vxpose"):
        return "xlu", int(re.search(r"\.xlu(\d+)", mnemonic).group(1)), "xpose"
    m = re.search(r"\.(mxu|xlu)(\d+)$", mnemonic)
    if m:
        cls, uid = m.group(1), int(m.group(2))
        if mnemonic.startswith("vpop"):
            return cls, uid, "pop"
        if mnemonic.startswith("vmatpush"):
            return cls, uid, "push"
        return cls, uid, "work"
    if mnemonic == "vpop.eup":
        return "eup", 0, "pop"
    if mnemonic.startswith(("vexp", "vlog", "vrcp", "vrsqrt", "vtanh", "vpow", "vsigmoid")):
        return "eup", 0, "work"
    return None


def _mclass(mnemonic):
    """Mnemonic without unit index, for attribution ('vpop.f32.mrf.mxu2' -> 'vpop.f32.mrf')."""
    return re.sub(r"\.(mxu|xlu)\d+$", "", mnemonic)


def _work_key(cls, mnemonic):
    """Sub-key selecting per-op parameters inside a unit class."""
    if cls == "mxu":
        return "bf16" if ".bf16" in mnemonic else "f32"
    if cls == "xlu":
        return "xlane" if ".xlane" in mnemonic else "permute"
    return "default"


class Units:
    def __init__(self, params):
        self.p = params
        self.busy = {}
        self.fifo = {}
        self.push_busy = {}
        self.staged = {}
        self.xcount = {}  # vregs pushed into the transpose in progress, per unit

    def _wp(self, cls, mnemonic):
        up = self.p[cls]
        return up["work"][_work_key(cls, mnemonic)]

    @staticmethod
    def _fifo_key(cls, uid, mnemonic):
        return (cls, uid, "trf") if ("xpose" in mnemonic or ".trf" in mnemonic) else (cls, uid)

    def ready(self, cls, uid, role, mnemonic, t):
        key = (cls, uid)
        if role == "xpose":
            return max(t, self.busy.get((cls, uid, "trf"), 0.0)) if self.p[cls].get("xpose") else t
        if role == "work":
            t = max(t, self.busy.get(key, 0.0))
            msr = re.search(r"\.vlgmr\.(msr\w)", mnemonic)
            if msr:
                t = max(t, self.staged.get((key, msr.group(1)), 0.0) + self.p[cls]["load_latency"])
            return t
        if role == "push":
            return max(t, self.push_busy.get(key, 0.0))
        if self.p[cls].get("nonblocking_pop"):
            return t
        q = self.fifo.get(self._fifo_key(cls, uid, mnemonic))
        return max(t, q[0]) if q else t

    def commit(self, cls, uid, role, mnemonic, t):
        """Returns, for a pop, the time its result reaches the register file (else None)."""
        key = (cls, uid)
        if role == "xpose":
            xps = self.p[cls].get("xpose")
            if not xps:  # uncalibrated: transposes cost only their issue slots
                return None
            xp = xps["b16bin" if ".binary" in mnemonic else "b16" if ".b16" in mnemonic else "b32"]
            n = 2 if ".binary" in mnemonic else 1
            tk = (cls, uid, "trf")
            self.busy[tk] = t + xp["occupancy"] * n
            self.xcount[tk] = (0 if ".start" in mnemonic else self.xcount.get(tk, 0)) + n
            if ".end" in mnemonic:
                q = self.fifo.setdefault(tk, [])
                q.extend(t + xp["latency"] + j * xp["interval"] for j in range(self.xcount[tk]))
            return None
        if role == "work":
            wp = self._wp(cls, mnemonic)
            self.busy[key] = t + wp["occupancy"]
            q = self.fifo.setdefault(key, [])
            q.extend(t + wp["latency"] + j * wp.get("interval", 0) for j in range(wp.get("results", 1)))
        elif role == "push":
            up = self.p[cls]
            if ".xpose" in mnemonic:
                cost = up.get("push_cycles_xpose", up["push_cycles"])
            elif ".bf16" in mnemonic:
                cost = up["push_cycles"]
            else:
                cost = up.get("push_cycles_f32", up["push_cycles"])
            done = t + cost
            self.push_busy[key] = done
            msr = re.search(r"\.(msr\w)\.", mnemonic).group(1)
            self.staged[(key, msr)] = done
        else:
            q = self.fifo.get(self._fifo_key(cls, uid, mnemonic))
            head = q.pop(0) if q else t
            return max(t, head) + self.p[cls]["pop_latency"]
        return None


def slot_of(m):
    """Issue queue of a vector op; None for scalar-side ops (run by the front end)."""
    if m.startswith("vpop") and m != "vpop.sfrf":
        return "pop"
    if m.startswith(("vmatmul", "vmatpush")):
        return "mxu"
    if re.search(r"\.xlu\d+$", m) or m.startswith("vxpose"):
        return "xlu"
    if m.startswith("vld"):
        return "vld"
    if m.startswith("vst"):
        return "vst"
    if m.startswith(("vexp", "vlog", "vrcp", "vrsqrt", "vtanh", "vpow", "vsigmoid")):
        return "eup"
    if m.startswith("v") and not m.startswith(("vsync", "vnop")):
        return "valu"
    return None


@dataclass
class _Bundle:
    groups: dict  # slot -> [(vreg srcs, unit op or None, (dest, latency, producer) or None, dest reg)]
    scalar_srcs: tuple  # (sreg, extra lag)
    scalar_dests: tuple  # (sreg, latency, producer)
    dma_start: bool


class Scoreboard:
    """Vector ops wait in per-slot in-order queues (pop, mxu, xlu, vld, vst, eup, valu); a slot starts
    one bundle's ops per cycle once their registers and units are ready. The scalar front end
    dispatches a bundle at most `issue_lead` cycles before its latest-starting vector op
    (issue_lead 0 = whole bundles issue in order) and stalls on scalar dependencies; a DMA start
    waits for queued vector stores."""

    def __init__(self, params):
        self.p = params
        self.lead = params.issue_lead
        self.ready_at = {}
        self.producer = {}  # reg -> mnemonic class of its last writer (stall attribution)
        self.stall_by = {}  # cause -> front-end stall cycles
        self.cache = {}
        self.units = Units(params.units)
        self.slot_free = {}
        self.done_max = 0.0
        self.pending = None
        # Scalar timing, off unless calibrated: `sld` = load-to-use latency; `saddr` = extra cycles
        # before a freshly written scalar register can serve as an sld/sst address.
        self.sld_latency = params.latency.get("sld", 1)
        self.saddr = params.latency.get("saddr", 0)
        self.scalar = "sld" in params.latency or "saddr" in params.latency

    def info(self, b):
        inf = self.cache.get(b.index)
        if inf is None:
            groups, ssrcs, sdests = {}, set(), []
            for op in b.ops:
                m = op.mnemonic
                slot = slot_of(m)
                if self.scalar:
                    # address generation: sld/sst/vld/vst address operands and scalar_lea inputs
                    is_lea = m.startswith("scalar_lea")
                    for a in op.args:
                        addr = is_lea or (a.lstrip().startswith("[") and (slot in ("vld", "vst") or m.startswith(("sld", "sst"))))
                        if slot is None or addr:
                            ssrcs.update((r, self.saddr if addr else 0) for r in SREG.findall(a))
                if slot is None:
                    if self.scalar:
                        if op.dest_reg and op.dest_reg[0] == "s":
                            sdests.append((op.dest_reg, self.sld_latency if m.startswith("sld") else 1, _mclass(m)))
                    continue
                srcs = set()
                for a in op.args:
                    srcs.update(VREG.findall(a))
                c = classify(m)
                unit = (*c, m) if c and c[0] in self.p.units else None
                dest = None
                if op.dest_reg and op.dest_reg[0] == "v":
                    if unit and unit[2] == "pop":
                        lat = self.p.units[unit[0]]["pop_latency"]
                    else:
                        lat = _longest_prefix(m, self.p.latency, self.p.default_latency)
                    dest = (op.dest_reg, lat, _mclass(m))
                groups.setdefault(slot, []).append((tuple(srcs), unit, dest, op.dest_reg))
            dma = any(op.mnemonic.startswith("dma.") and op.mnemonic != "dma.done.wait" for op in b.ops)
            inf = self.cache[b.index] = _Bundle(groups, tuple(ssrcs), tuple(sdests), dma)
        return inf

    def issue_time(self, b, t):
        inf = self.info(b)
        t0, cause = t, None
        for r, lag in inf.scalar_srcs:
            tr = self.ready_at.get(r, 0.0) + lag
            if tr > t:
                t, cause = tr, "wait " + self.producer.get(r, "?")
        if inf.dma_start and self.slot_free.get("vst", 0.0) > t:
            t, cause = self.slot_free["vst"], "dma start waits for vector stores"
        floor, latest, latest_cause = {}, t, None
        for slot, ops in inf.groups.items():
            g, c = max(t, self.slot_free.get(slot, 0.0)), None
            for srcs, unit, dest, _ in ops:
                for r in srcs:
                    tr = self.ready_at.get(r, 0.0)
                    if tr > g:
                        g, c = tr, "wait " + self.producer.get(r, "?")
                if unit:
                    tu = self.units.ready(*unit, g)
                    if tu > g:
                        g, c = tu, f"unit {unit[0]} {unit[2]} ({_mclass(unit[3])})"
            floor[slot] = g
            if g > latest:
                latest, latest_cause = g, c
        d = max(t, latest - self.lead)
        if d > t:
            cause = latest_cause
        if d > t0:
            self.stall_by[cause] = self.stall_by.get(cause, 0.0) + d - t0
        self.pending = {slot: max(d, g) for slot, g in floor.items()}
        return d

    def commit(self, b, t):
        inf = self.info(b)
        for r, lat, m in inf.scalar_dests:
            self.ready_at[r] = t + lat
            self.producer[r] = m
        for slot, ops in inf.groups.items():
            g = self.pending[slot]
            self.slot_free[slot] = g + 1
            self.done_max = max(self.done_max, g + 1)
            for srcs, unit, dest, dreg in ops:
                if dest:
                    self.ready_at[dest[0]] = g + dest[1]
                    self.producer[dest[0]] = dest[2]
                    self.done_max = max(self.done_max, g + dest[1])
                if unit:
                    done = self.units.commit(*unit, g)
                    if done is not None and dreg:
                        self.ready_at[dreg] = max(self.ready_at.get(dreg, 0.0), done)
                        self.done_max = max(self.done_max, done)

    def drain(self):
        """Cycle by which every queued vector op has finished."""
        return self.done_max
