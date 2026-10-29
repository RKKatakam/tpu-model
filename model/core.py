"""Issue-stall model for the vector side: a register scoreboard plus FIFO units.

A bundle issues at the first cycle where every vector register it reads is ready and
every unit it needs can accept work (in-order VLIW with interlocks, HARDWARE_FACTS F9).

FIFO units (MXU, XLU, EUP) take work ops and return results through `vpop.*`:
- work op on unit u starts when u is free, occupies it `occupancy` cycles, and queues
  `results` entries that become poppable `latency` (+ j * `interval`) cycles after start;
- a pop waits for its unit's oldest entry; the popped vreg is ready `pop_latency` later.
MXU weights: `vmatpush.<msr>` streams into staging register msr (one push path per MXU,
`push_cycles` each, `push_cycles_f32` for f32 weights, which take two pushes per bf16 one); a
`vmatmul.vlgmr.<msr>` first waits until those pushes finish plus `load_latency`.
"""

import re
from dataclasses import dataclass, field

VREG = re.compile(r"%\w+?_(vm\d+|v\d+)\b")


@dataclass
class CoreParams:
    latency: dict = field(default_factory=dict)  # mnemonic prefix -> result latency (cycles)
    default_latency: int = 1
    units: dict = field(default_factory=dict)  # unit class -> params, see classify()


def _longest_prefix(mnemonic, table, default):
    best, best_len = default, -1
    for prefix, val in table.items():
        if mnemonic.startswith(prefix) and len(prefix) > best_len:
            best, best_len = val, len(prefix)
    return best


def classify(mnemonic):
    """(unit class, unit id, role) for FIFO-unit ops, else None.

    role: 'work' (produces results), 'pop', 'push' (MXU weight staging)."""
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


def _work_key(cls, mnemonic):
    """Sub-key selecting per-op parameters inside a unit class."""
    if cls == "mxu":
        return "bf16" if ".bf16" in mnemonic else "f32"
    if cls == "xlu":
        return "xlane" if ".xlane" in mnemonic else "permute"
    return "default"


@dataclass
class _Info:
    srcs: tuple
    dests: tuple  # (vreg, latency, is_pop)
    unit_ops: tuple  # (cls, uid, role, mnemonic)


class Units:
    def __init__(self, params):
        self.p = params
        self.busy = {}
        self.fifo = {}
        self.push_busy = {}
        self.staged = {}

    def _wp(self, cls, mnemonic):
        up = self.p[cls]
        return up["work"][_work_key(cls, mnemonic)]

    def ready(self, cls, uid, role, mnemonic, t):
        key = (cls, uid)
        if role == "work":
            t = max(t, self.busy.get(key, 0.0))
            msr = re.search(r"\.vlgmr\.(msr\w)", mnemonic)
            if msr:
                t = max(t, self.staged.get((key, msr.group(1)), 0.0) + self.p[cls]["load_latency"])
            return t
        if role == "push":
            return max(t, self.push_busy.get(key, 0.0))
        q = self.fifo.get(key)
        return max(t, q[0]) if q else t

    def commit(self, cls, uid, role, mnemonic, t):
        key = (cls, uid)
        if role == "work":
            wp = self._wp(cls, mnemonic)
            self.busy[key] = t + wp["occupancy"]
            q = self.fifo.setdefault(key, [])
            q.extend(t + wp["latency"] + j * wp.get("interval", 0) for j in range(wp.get("results", 1)))
        elif role == "push":
            up = self.p[cls]
            done = t + (up["push_cycles"] if ".bf16" in mnemonic else up.get("push_cycles_f32", up["push_cycles"]))
            self.push_busy[key] = done
            msr = re.search(r"\.(msr\w)\.", mnemonic).group(1)
            self.staged[(key, msr)] = done
        else:
            q = self.fifo.get(key)
            if q:
                q.pop(0)


class Scoreboard:
    def __init__(self, params):
        self.p = params
        self.ready_at = {}
        self.cache = {}
        self.units = Units(params.units)

    def info(self, b):
        inf = self.cache.get(b.index)
        if inf is None:
            srcs, dests, unit_ops = set(), [], []
            for op in b.ops:
                for a in op.args:
                    srcs.update(VREG.findall(a))
                c = classify(op.mnemonic)
                if c and c[0] in self.p.units:
                    unit_ops.append((*c, op.mnemonic))
                if op.dest_reg and op.dest_reg[0] == "v":
                    if c and c[2] == "pop" and c[0] in self.p.units:
                        lat = self.p.units[c[0]]["pop_latency"]
                    else:
                        lat = _longest_prefix(op.mnemonic, self.p.latency, self.p.default_latency)
                    dests.append((op.dest_reg, lat))
            inf = self.cache[b.index] = _Info(tuple(srcs), tuple(dests), tuple(unit_ops))
        return inf

    def issue_time(self, b, t):
        inf = self.info(b)
        for r in inf.srcs:
            t = max(t, self.ready_at.get(r, 0.0))
        for cls, uid, role, m in inf.unit_ops:
            t = max(t, self.units.ready(cls, uid, role, m, t))
        return t

    def commit(self, b, t):
        inf = self.info(b)
        for r, lat in inf.dests:
            self.ready_at[r] = t + lat
        for cls, uid, role, m in inf.unit_ops:
            self.units.commit(cls, uid, role, m, t)
