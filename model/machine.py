"""Bundle-level replay: scalar interpreter + discrete-event timing.

The scalar core is executed for real (registers, predicates, SMEM, sync flags),
so loop trip counts, predicated DMAs, and branch targets come from the program
itself. Vector and MXU ops only cost their bundle's issue slot. DMAs are timed
by a pluggable DmaModel; `dma.done.wait` stalls until its flag has received the
awaited granules.

Semantics assumed here (hypotheses until confirmed, see parser/MNEMONICS.md):
- one bundle issues per cycle; ops in a bundle read state before any op writes;
- `sbr.rel` takes effect after BRANCH_DELAY following bundles;
- a DMA adds size_in_granules to its dst_syncflagno flag on completion;
- `dma.done.wait flag, n` blocks until flag >= n; `vsyncadd flag, d` adds d.
"""

import re
import struct
from dataclasses import dataclass, field

BRANCH_DELAY = 4
GRANULE_BYTES = 32
MASK = 0xFFFFFFFF

MEM = re.compile(r"^\[(?:(\w+):)?\[?(.*?)\]?(?:\s+sm:\S+)?\]$")


def u32(x):
    return x & MASK


def s32(x):
    x &= MASK
    return x - (1 << 32) if x & 0x80000000 else x


@dataclass
class Dma:
    op_index: tuple[int, int]
    kind: str
    granules: int
    flag: int
    issue: float
    done: float = 0.0
    descriptor: str = ""


@dataclass
class Trace:
    cycles: float = 0.0
    bundles_issued: int = 0
    stall_cycles: float = 0.0
    core_stall_cycles: float = 0.0
    stalls: dict = field(default_factory=dict)
    dmas: list = field(default_factory=list)
    halts: list = field(default_factory=list)
    unmodeled_smem_dmas: int = 0
    waits: list = field(default_factory=list)  # (bundle index, issue cycle, flag, stall cycles) per dma.done.wait


class Machine:
    # Operand registers hold addresses in 512 B units (gmm's bounds check: an 8 MiB operand spans
    # 16384); kernels shift them left by 4 to get 32 B-granule DMA pointers and check them as
    # signed. Operand i sits at (i + 1) << 20 units: 512 MiB apart, shifted pointers stay positive.
    OPERAND_SHIFT = 20
    WORDS_PER_GRANULE = 8  # 32 B DMA granules (hbm_to_smem moves 16 for s32[8], s32[16] and s32[32])

    def __init__(self, program, operand_bases=None, smem=None, max_bundles=50_000_000, core=None, operands=None):
        self.p = program
        self.core = core
        self.regs = {}
        self.smem = dict(smem or {})
        self.flags = {}
        self.sfrf = []
        self.drf = []
        self.alloc = {}
        self.operand_bases = dict(operand_bases or {})
        self.max_bundles = max_bundles
        # integer operands (model/operands.py): scalars sit in their register, arrays in memory
        self.data = {}
        for idx, (space, v) in (operands or {}).items():
            if space == "scalar":
                self.operand_bases[idx] = v
            else:
                self.data[idx + 1] = v

    def data_words(self, ptr):
        """Words of the integer operand a 32 B-granule DMA pointer points into."""
        key = ptr >> (self.OPERAND_SHIFT + 4)
        if key in self.data:
            return self.data[key][(ptr & ((1 << (self.OPERAND_SHIFT + 4)) - 1)) * 8:]
        return None

    def copy_to_smem(self, op, granules, trace):
        src = self.val(op.labels.get("hbm") or op.labels["vmem"])
        src_words = self.data_words(src)
        if src_words is None:
            trace.unmodeled_smem_dmas += 1
            return
        dst = self.val(op.labels["smem"])
        for j, w in enumerate(src_words[: granules * self.WORDS_PER_GRANULE]):
            self.smem[dst + j] = w

    def addr_of(self, name):
        if name not in self.alloc:
            self.alloc[name] = (len(self.alloc) + 1) << 16
        return self.alloc[name]

    def val(self, tok):
        # `scalar_lea.vmem %s, 16 [#allocation3]`: trailing [#allocation] names the region; not an operand
        tok = re.sub(r"\s+\[#[\w.]+\]$", "", tok.strip())
        if tok.startswith("%"):
            reg = re.search(r"_((?:s|p)\d+)$", tok)
            if not reg:
                raise KeyError(f"not a scalar/predicate register: {tok}")
            return self.regs.get(reg.group(1), 0)
        if tok.startswith("["):
            return self.mem_addr(tok)
        if tok.startswith("#"):
            return self.addr_of(tok)
        return int(tok, 0)

    def mem_addr(self, tok):
        inner = tok
        while inner.startswith("["):
            inner = inner[1:-1] if inner.endswith("]") else inner[1:]
            inner = re.sub(r"^\w+:", "", inner).strip()
            inner = re.sub(r"\s+sm:\S+$", "", inner).strip()
        inner = inner.strip("[]")
        total = 0
        for part in inner.split("+"):
            part = part.strip()
            if part.startswith("$"):
                total += int(part[1:], 0)
            elif part.startswith("#"):
                total += self.addr_of(part)
            elif part:
                total += self.val(part)
        return total

    def vmem_addrs(self, b):
        """Addresses of this bundle's vector loads/stores (op index -> address); skipped if unparsable."""
        out = {}
        for j, op in enumerate(b.ops):
            if op.mnemonic.startswith(("vld", "vst")) and not op.mnemonic.startswith("vstv") and op.args and self.pred_ok(op):
                try:
                    out[j] = self.mem_addr(op.args[0])
                except (KeyError, ValueError):
                    pass
        return out

    def pred_ok(self, op):
        if op.pred is None:
            return True
        neg, reg = op.pred
        v = bool(self.regs.get(re.search(r"(p\d+)$", reg).group(1), 0))
        return v != neg

    def scalar(self, op):
        """Return the value an s/p-writing op produces, or None for no write."""
        if op.cond is not None:
            neg, preg, fallback = op.cond
            if bool(self.regs.get(re.search(r"(p\d+)$", preg).group(1), 0)) == neg:
                return self.val(fallback)
        m, a = op.mnemonic, op.args
        v = self.val
        base = m.split(".")[0]
        if base == "smov":
            return v(a[-1])
        if base == "sphi":
            return None
        if base in ("sadd", "ssub", "smul", "sand", "sor", "sxor", "sshll", "sshrl", "sshra"):
            x, y = v(a[0]), v(a[1])
            return u32(
                {
                    "sadd": lambda: x + y,
                    "ssub": lambda: x - y,
                    "smul": lambda: s32(x) * s32(y),
                    "sand": lambda: x & y,
                    "sor": lambda: x | y,
                    "sxor": lambda: x ^ y,
                    "sshll": lambda: x << (y & 31),
                    "sshrl": lambda: u32(x) >> (y & 31),
                    "sshra": lambda: s32(x) >> (y & 31),
                }[base]()
            )
        if base == "scvt":  # scalar int <-> float conversion; values are kept as 32-bit patterns
            src, dst = m.split(".")[1], m.split(".")[2]
            x = v(a[-1])
            f = struct.unpack("<f", struct.pack("<I", u32(x)))[0] if src == "f32" else s32(x)
            if dst == "f32":
                return struct.unpack("<I", struct.pack("<f", float(f)))[0]
            return u32(int(f))
        if base in ("smin", "smax"):
            x, y = v(a[0]), v(a[1])
            if m.split(".")[1].startswith("s"):
                x, y = s32(x), s32(y)
            return u32(min(x, y) if base == "smin" else max(x, y))
        if base == "scmp":
            cond, ty = m.split(".")[1], m.split(".")[2]
            x, y = v(a[0]), v(a[1])
            if ty.startswith("s"):
                x, y = s32(x), s32(y)
            else:
                x, y = u32(x), u32(y)
            return int({"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y}[cond])
        if base in ("por", "pand", "pnand", "pnor", "pxor"):
            x, y = bool(v(a[0])), bool(v(a[1]))
            return int({"por": x or y, "pand": x and y, "pnand": not (x and y), "pnor": not (x or y), "pxor": x != y}[base])
        if base == "pneg":
            return int(not v(a[0]))
        if base == "pmov":
            return int(bool(v(a[0])))
        if base == "scalar_select":
            return v(a[1]) if v(a[0]) else v(a[2])
        if base == "scalar_lea":
            return u32(v(a[0]) + v(a[1]))
        if base == "sld":
            return self.smem.get(self.mem_addr(a[0]), 0)
        if m == "vpop.sfrf":
            return self.sfrf.pop(0)
        if m == "spop.drf":  # divide result FIFO: quotient, then remainder (sdivrem)
            return self.drf.pop(0)
        if base == "inlined_call_operand":
            idx = int(re.search(r"index:\s*(\d+)", a[0]).group(1))
            return self.operand_bases.get(idx, (idx + 1) << self.OPERAND_SHIFT)
        raise NotImplementedError(f"scalar op {m}: {op.text}")

    def bind_operands(self):
        """inlined_call_operand ops define register values at kernel entry, before bundle 0 issues."""
        for b in self.p.bundles:
            for op in b.ops:
                if op.mnemonic.startswith("inlined_call_operand"):
                    self.regs[op.dest_reg] = self.scalar(op)

    def run(self, dma_model):
        self.bind_operands()
        bundles, trace = self.p.bundles, Trace()
        pc, t, branch = 0, 0.0, None
        inflight = []
        while pc < len(bundles):
            if trace.bundles_issued >= self.max_bundles:
                raise RuntimeError("bundle budget exceeded; runaway loop?")
            b = bundles[pc]
            for op in b.ops:
                if op.mnemonic == "dma.done.wait" and self.pred_ok(op):
                    flag, need = self.val(op.args[0]), self.val(op.args[1])
                    dma_model.project(t)
                    ready = self._flag_ready_time(inflight, flag, need, t)
                    if ready > t:  # a core that actually blocks pays the semaphore wake-up
                        ready += getattr(dma_model, "wake_cycles", 0.0)
                    trace.waits.append((pc, t, flag, max(0.0, ready - t)))
                    if ready > t:
                        key = (pc, op.comment[:40])
                        trace.stalls[key] = trace.stalls.get(key, 0.0) + (ready - t)
                        trace.stall_cycles += ready - t
                        t = ready
                    self._retire(inflight, t)
            if self.core is not None:
                if self.core.vmem_raw:
                    self.core.mem_addrs = self.vmem_addrs(b)
                t_issue = self.core.issue_time(b, t)
                trace.core_stall_cycles += t_issue - t
                t = t_issue
            writes, new_branch, pseudo = [], None, []
            for j, op in enumerate(b.ops):
                if not self.pred_ok(op):
                    continue
                m = op.mnemonic
                if m.startswith("inlined_call_operand"):
                    continue
                if m.startswith("int_to_ptr"):
                    pseudo.append(op)
                elif op.dest_reg and op.dest_reg[0] in "sp" and not op.dest_reg.startswith("vm"):
                    r = self.scalar(op)
                    if r is not None:
                        writes.append((op.dest_reg, r))
                elif m == "sst":
                    writes.append((("smem", self.mem_addr(op.args[0])), self.val(op.args[1])))
                elif m == "vsyncpa":
                    writes.append((("flag", self.val(op.args[0])), self.val(op.args[1])))
                elif m.startswith("sdivrem"):
                    x, y = self.val(op.args[0]), self.val(op.args[1])
                    if m.endswith(".s32"):
                        x, y = s32(x), s32(y)
                        q = int(x / y) if y else 0
                        self.drf += [u32(q), u32(x - q * y)]
                    else:
                        self.drf += [u32(x // y) if y else 0, u32(x % y) if y else 0]
                elif m == "vsyncmov":
                    self.sfrf.append(self.flags.get(self.val(op.args[0]), 0))
                elif m == "vsyncadd":
                    writes.append((("flagadd", self.val(op.args[0])), s32(self.val(op.args[1]))))
                elif m.startswith("dma.") and m != "dma.done.wait":
                    d = Dma((pc, j), m, self.val(op.labels["size_in_granules"]),
                            self.val(op.labels["dst_syncflagno"]), t, descriptor=op.comment)
                    if m.endswith("_to_smem"):
                        self.copy_to_smem(op, d.granules, trace)
                    dma_model.issue(d, t)
                    inflight.append(d)
                    trace.dmas.append(d)
                elif m == "sbr.rel":
                    target = int(re.search(r"\$region(\d+)", op.args[0]).group(1))
                    new_branch = [self.p.target(target), BRANCH_DELAY]
                elif m == "shalt.err":
                    trace.halts.append(pc)
            # int_to_ptr is a free cast that reads its source after same-bundle writes.
            fwd = {k: v for k, v in writes if not isinstance(k, tuple)}
            for op in pseudo:
                src = re.search(r"_((?:s|p)\d+)$", op.args[-1]).group(1)
                writes.append((op.dest_reg, fwd.get(src, self.regs.get(src, 0))))
            for key, val in writes:
                if isinstance(key, tuple):
                    kind, addr = key
                    if kind == "smem":
                        self.smem[addr] = val
                    elif kind == "flag":
                        self.flags[addr] = val
                    else:
                        self.flags[addr] = self.flags.get(addr, 0) + val
                else:
                    self.regs[key] = val
            if self.core is not None:
                self.core.commit(b, t)
            t += 1
            trace.bundles_issued += 1
            if branch is not None:
                branch[1] -= 1
                if branch[1] == 0:
                    pc, branch = branch[0], None
                    continue
            if new_branch is not None:
                branch = new_branch
            pc += 1
        dma_model.project(t)
        trace.cycles = max([t] + [d.done for d in inflight] + ([self.core.drain()] if self.core is not None else []))
        return trace

    def _flag_ready_time(self, inflight, flag, need, t):
        """Earliest time >= t at which flag + granules of DMAs landed by then reaches need."""
        have = self.flags.get(flag, 0)
        if have >= need:
            return t
        for d in sorted((d for d in inflight if d.flag == flag), key=lambda d: d.done):
            have += d.granules
            if have >= need:
                return max(t, d.done)
        raise RuntimeError(f"wait on flag {flag:#x} for {need} can never be satisfied")

    def _retire(self, inflight, t):
        for d in [d for d in inflight if d.done <= t]:
            self.flags[d.flag] = self.flags.get(d.flag, 0) + d.granules
            inflight.remove(d)
