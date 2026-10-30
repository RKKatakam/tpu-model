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
    elif kernel in ("dma_out", "dma_conc", "dma_inout", "dma_strided"):
        starts = collections.Counter(
            (op.mnemonic, op.labels.get("size_in_granules"))
            for b in program.bundles
            for op in b.ops
            if op.mnemonic.startswith("dma.") and op.mnemonic != "dma.done.wait"
        )
        g = params["rows"] * 128 * 4 // 32
        want = {
            "dma_out": {("dma.vmem_to_hbm", str(g)): params["reps"]},
            "dma_conc": {("dma.hbm_to_vmem", str(g)): params["reps"] * params.get("streams", 1)},
            "dma_inout": {("dma.hbm_to_vmem", str(g)): params["reps"], ("dma.vmem_to_hbm", str(g)): params["reps"]},
            "dma_strided": {("dma.hbm_to_vmem", str(g)): params["reps"]},
        }[kernel]
        expect(dict(starts) == want, f"dma starts {dict(starts)} want {want}")
        vec = {m: n for m, n in c.items() if m.startswith("v") and not m.startswith("vsync") and m != "vpop.sfrf"}
        expect(not vec, f"vector ops {vec}")
    elif kernel == "mxu_dot":
        lhs = params.get("lhs", "bf16")
        mm = {m for m in c if m.startswith("vmatmul.")}
        expect(mm and all(m.startswith(f"vmatmul.{lhs}.") for m in mm), f"matmul ops {sorted(mm)} want vmatmul.{lhs}.*")
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
        expect(not has_spill(program), "spills")
    elif kernel == "unit_chain":
        mm = sum(v for m, v in c.items() if m.startswith("vmatmul."))
        xl = sum(v for m, v in c.items() if m.startswith("vmax.xlane") or ".xlane" in m and not m.startswith("vpop"))
        ex = sum(v for m, v in c.items() if m.startswith("vexp"))
        if "mxu" in params["kind"]:
            expect(mm >= params["k"], f"vmatmul={mm} want >= {params['k']}")
        if "xlu" in params["kind"]:
            expect(xl >= params["k"], f"xlane ops={xl} want >= {params['k']}")
        if "eup" in params["kind"]:
            expect(ex >= params["k"], f"vexp={ex} want >= {params['k']}")
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
        expect(not has_spill(program), "spills")
    elif kernel == "dma_compute":
        g = params["block_rows"] * 1024 * 4 // 32
        starts = {(op.mnemonic, op.labels.get("size_in_granules")) for b in program.bundles for op in b.ops
                  if op.mnemonic.startswith("dma.") and op.mnemonic != "dma.done.wait"}
        want = set()
        if params.get("reads", 1):
            want.add(("dma.hbm_to_vmem", str(g)))
        want.add(("dma.vmem_to_hbm", str(g) if params.get("writes", 1) else str(8 * 1024 * 4 // 32)))
        expect(starts == want, f"dma starts {sorted(starts)} want {sorted(want)}")
        if params["kind"] == "valu":
            expect(c["vadd.f32"] >= params["amount"], f"vadd.f32={c['vadd.f32']}")
        if params["kind"] == "vmem":
            vst = sum(v for m, v in c.items() if m.startswith("vst"))
            expect(vst >= params["amount"] * 1024 // 1024, f"vst={vst}: VMEM copy eliminated?")
    elif kernel == "kv_strided":
        if params["step"] > 1:
            expect(c["vld.sshfl"] >= 1 or any(m.startswith("vld.") for m in c), f"no strided loads: {sorted(m for m in c if m.startswith('vld'))}")
        expect(c["sbr.rel"] >= 1, "no loop")
    elif kernel == "tgmm_tile":
        msk = sum(v for m, v in c.items() if m.startswith("vmatpush") and ".msk" in m)
        if params["masked"]:
            expect(msk >= 1, f"no masked pushes: {sorted(m for m in c if m.startswith('vmatpush'))}")
        expect(not {m for m in dma_ops(c) if not m.endswith("_to_smem")} - {"dma.vmem_to_hbm", "dma.done.wait"}, f"dma {dma_ops(c)}")
    elif kernel == "vmem_ops":
        want = {"vst_msk": "vst.msk", "vld_sshfl": "vld.sshfl", "vst_sshfl": "vst.sshfl", "vrot_slane": "vrot.slane"}.get(params["form"])
        if want:
            expect(c[want] >= params["n"], f"{want}={c[want]} want >= {params['n']}: {sorted(m for m in c if m.startswith(('vld', 'vst', 'vrot')))}")
        expect(c["sbr.rel"] >= 1, "no loop")
    elif kernel == "vmem_access":
        want = {"stride2": "vld.sshfl", "stride4": "vld.sshfl", "stride2_store": "vst.sshfl", "masked": "vst.msk"}.get(params["form"])
        if want:
            expect(c[want] >= 1, f"no {want}: {sorted(m for m in c if m.startswith(('vld', 'vst')))}")
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
    elif kernel == "vop_chain2":
        vec = sum(v for m, v in c.items() if m.startswith("v") and not m.startswith(("vld", "vst", "vsync", "vnop", "vpop.sfrf")))
        expect(vec >= params["k"] * params["nv"], f"vector ops {vec} < k*nv={params['k'] * params['nv']} (folded?)")
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
        expect(not has_spill(program), "spills")
    elif kernel == "attn_step":
        expect(sum(v for m, v in c.items() if m.startswith("vmatmul")) >= 1, "no vmatmul")
        if params.get("part", "full") not in ("qk", "pv"):
            expect(any(m.startswith("vpop.xlane") for m in c), "no cross-lane reduction")
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
    elif kernel == "dma_mxu":
        reads = c["dma.hbm_to_vmem"]
        expect(reads >= params["n_in"], f"hbm_to_vmem starts {reads}")
        if params["work"] == "mxu":
            expect(sum(v for m, v in c.items() if m.startswith("vmatmul")) >= 1, "no vmatmul")
    elif kernel == "sld_unrolled":
        expect(c["sld"] >= params["unroll"], f"sld={c['sld']} want >= {params['unroll']}")
        expect((c["sbr.rel"] == 0) == (params["unroll"] == params["n"]), f"branches={c['sbr.rel']}")
    elif kernel == "pop_loop":
        pop = "vpop.f32.mrf" if params["kind"] == "mxu" else "vpop.xlane"
        expect(any(m.startswith(pop) for m in c), f"no {pop}: {sorted(m for m in c if m.startswith('vpop'))}")
        if params.get("step", "mul").startswith("mul"):
            expect(sum(v for m, v in c.items() if m.startswith("smul")) >= 1, "no scalar multiply")
        expect(c["sbr.rel"] >= 2, f"branches={c['sbr.rel']} (want outer + inner loop)")
    elif kernel == "pop_scalar":
        pop = "vpop.f32.mrf" if params["kind"] == "mxu" else "vpop.xlane"
        expect(any(m.startswith(pop) for m in c), f"no {pop}: {sorted(m for m in c if m.startswith('vpop'))}")
        smul = sum(v for m, v in c.items() if m.startswith("smul"))
        expect(smul >= params["n"], f"scalar multiplies {smul} < n={params['n']} (folded?)")
        expect(c["sbr.rel"] >= (2 if params["n"] else 1), f"branches={c['sbr.rel']} (want loop + region)")
        expect(not {m for m in dma_ops(c) if not m.endswith("_to_smem")} - {"dma.vmem_to_hbm", "dma.done.wait"},
               f"dma ops {dma_ops(c)}")
    elif kernel == "xpose_chain":
        n = sum(v for m, v in c.items() if m.startswith("vxpose"))
        expect(n >= params["k"] * params["nv"], f"vxpose ops {n} want >= {params['k'] * params['nv']}")
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
    elif kernel == "mxu_masked":
        expect(any(".msk" in m for m in c if m.startswith("vmatpush")), f"no masked pushes: {sorted(m for m in c if m.startswith('vmatpush'))}")
    elif kernel == "mxu_fresh":
        expect(any(m.startswith("vmatpush") for m in c), "no weight pushes")
        expect(not dma_ops(c), f"unexpected DMA ops {dma_ops(c)}")
        expect(not has_spill(program), "spills")
    elif kernel == "vmem_chain":
        expect(any(m.startswith("vld") for m in c) and any(m.startswith("vst") for m in c), "no vld/vst")
        expect(c["sbr.rel"] >= 1, "no loop")
    elif kernel == "sld_use":
        expect(c["sld"] >= 1 and c["sbr.rel"] >= 1, f"sld={c['sld']} branches={c['sbr.rel']}")
    elif kernel == "sld_chain":
        expect(not {m for m in dma_ops(c) if not m.endswith("_to_smem")} - {"dma.vmem_to_hbm", "dma.done.wait"},
               f"dma ops {dma_ops(c)}")
        expect(c["sld"] >= 1 and c["sbr.rel"] >= 1, f"sld={c['sld']} branches={c['sbr.rel']}")
    elif kernel == "dma_overhead":
        starts = {m for m in dma_ops(c) if m != "dma.done.wait"}
        expect(starts == ({"dma.hbm_to_vmem"} if params["dma"] else set()), f"dma starts {starts}")
        expect(c["vadd.f32"] == params["k"], f"vadd.f32={c['vadd.f32']} want {params['k']}")
        expect(not has_spill(program), "spills")
    elif kernel == "vop_chain" and params["nv"] >= 16:
        want = {"roll_lane": "vrot.lane", "sum_lane": "vadd.xlane"}[params["op"]]
        n = sum(v for m, v in c.items() if m.startswith(want))
        expect(n == params["k"] * params["nv"], f"{want} count {n} want {params['k'] * params['nv']}")
        expect(not has_spill(program), "spills")
    elif kernel == "dma_gather":
        g = params["rows"] * 128 * 4 // 32
        starts = {(op.mnemonic, op.labels.get("size_in_granules")) for b in program.bundles for op in b.ops
                  if op.mnemonic.startswith("dma.") and op.mnemonic != "dma.done.wait"
                  and not op.mnemonic.endswith("_to_smem")}
        want = {("dma.hbm_to_vmem", str(g)), ("dma.vmem_to_hbm", str(g))}
        expect(starts == want, f"dma starts {sorted(starts)} want {sorted(want)}")
        vec = {m for m in c if m.startswith("v") and not m.startswith(("vsync", "vld", "vst", "vpop.sfrf", "vmov", "vnop"))}
        expect(not vec, f"vector ops {sorted(vec)}")
    elif kernel == "dma_stream":
        g = params["block_rows"] * params["cols"] * 4 // 32
        starts = {(op.mnemonic, op.labels.get("size_in_granules")) for b in program.bundles for op in b.ops
                  if op.mnemonic.startswith("dma.") and op.mnemonic != "dma.done.wait"}
        want = set()
        if params["reads"]:
            want.add(("dma.hbm_to_vmem", str(g)))
        if params["writes"]:
            want.add(("dma.vmem_to_hbm", str(g)))
        else:
            want.add(("dma.vmem_to_hbm", str(8 * params["cols"] * 4 // 32)))  # resident (8, cols) output
        expect(starts == want, f"dma starts {sorted(starts)} want {sorted(want)}")
        vec = {m for m in c if m.startswith("v") and not m.startswith(("vsync", "vld", "vst", "vadd", "vpop.sfrf", "vmov", "vnop"))}
        expect(not vec, f"vector ops beyond vld/vst/vadd: {sorted(vec)}")
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
