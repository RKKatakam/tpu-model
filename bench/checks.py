"""Bundle-level checks that each microbenchmark exercises only what it claims (rule 5).

usage: python -m bench.checks RECORDS_DIR [RECORDS_DIR ...]
Prints one line per record and exits non-zero if any check fails.
"""

import collections
import sys

from model.predict import load_records

SPILL_HINTS = ("_spill", "VSTORE:SPILL")


def op_counts(program):
    c = collections.Counter()
    for b in program.bundles:
        for op in b.ops:
            c[op.mnemonic] += 1
    return c


def has_spill(program):
    return any("_spill" in op.text for b in program.bundles for op in b.ops)


def dma_ops(c):
    return {m: n for m, n in c.items() if m.startswith("dma.")}


def check(kernel, params, program):
    """Return a list of failed expectations (empty when the bundle dump is as intended)."""
    c, fails = op_counts(program), []

    def expect(cond, msg):
        if not cond:
            fails.append(msg)

    if kernel == "valu_straight":
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
        expect(c["sbr.rel"] == 0, "has branches")
        expect(c["vadd.f32"] == params["k"] * params["nv"], f"vadd.f32={c['vadd.f32']} want {params['k'] * params['nv']}")
        expect(not has_spill(program), "spills")
    elif kernel == "valu_loop":
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
        expect(c["vadd.f32"] == params["k"] * params["nv"], f"vadd.f32={c['vadd.f32']} want {params['k'] * params['nv']}")
        expect(c["sbr.rel"] >= 1 or params["trip"] == 1, "no loop branch")
        expect(not has_spill(program), "spills")
    elif kernel == "dma_seq":
        want = params["rows"] * 128 * 4 // 32
        starts = [
            (op.mnemonic, op.labels.get("size_in_granules"))
            for b in program.bundles
            for op in b.ops
            if op.mnemonic.startswith("dma.") and op.mnemonic != "dma.done.wait"
        ]
        expect(starts == [("dma.hbm_to_vmem", str(want))] * params["reps"], f"dma starts {starts[:3]}... want {params['reps']} x {want}")
        expect(c["dma.done.wait"] == params["reps"], f"waits={c['dma.done.wait']}")
        # The epilogue reads the semaphore back (vsyncmov + vpop.sfrf) and halts if it is non-zero.
        vec = {m: n for m, n in c.items() if m.startswith("v") and not m.startswith("vsync") and m != "vpop.sfrf"}
        expect(not vec, f"vector ops {vec}")
    return fails, c


def main(dirs):
    bad = 0
    for d in dirs:
        for r in load_records(d):
            kernel = r.raw["kernel"]
            if r.raw["kernel_class"] != "micro":
                continue
            p = r.program()
            fails, _ = check(kernel, r.raw["params"], p)
            bad += bool(fails)
            status = "ok  " if not fails else "FAIL"
            print(f"{status} {r.raw['record_id']:32s} bundles={len(p.bundles):6d} {'; '.join(fails)}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
