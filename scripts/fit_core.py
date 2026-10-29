"""Fit vector-side core parameters (latencies, EUP/XLU/MXU unit timings) from phase3a microbenchmarks.

usage: fit_core.py PHASE3A_DIR PHASE3D_DIR PHASE3F_DIR PHASE3E_DIR PHASE3J_DIR PHASE3N_DIR PHASE3P_DIR OUT_JSON
Each parameter family is fitted only on the microbenchmark that isolates it; MXU parameters are
fitted on reps=4 runs and checked on reps=16 runs.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.core import CoreParams, Scoreboard  # noqa: E402
from model.dma import NoDma  # noqa: E402
from model.machine import Machine  # noqa: E402
from model.predict import load_records  # noqa: E402

CLOCK_GHZ = 1.5  # HARDWARE_FACTS F7
LEAD = float(os.environ.get("TPUM_ISSUE_LEAD", "0"))  # front-end lead over queued vector ops



def cycles(record, params):
    from model.operands import integer_operands
    from model.predict import dma_model, load_dma

    prog = record.program()
    ops = integer_operands(record, prog) if record.raw["kernel"] in ("tgmm_tile", "sld_chain", "dma_gather") else None
    return Machine(prog, core=Scoreboard(params), operands=ops).run(dma_model(load_dma(), CLOCK_GHZ)).cycles


def measured_cycles(record):
    return record.measured_ns() * CLOCK_GHZ


def rel_errors(records, params):
    return np.array([cycles(r, params) / measured_cycles(r) - 1 for r in records])


def fit(objective_records, build, x0, names):
    def loss(x):
        return float(np.sum(rel_errors(objective_records, build(x)) ** 2))

    res = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 0.25, "fatol": 1e-7, "maxiter": 400})
    return dict(zip(names, res.x)), res


def main(batch_dir, f32_dir, f3_dir, gather_dir, xpose_dir, tgmm_dir, int8_dir, out_json):
    recs = load_records(batch_dir)
    vop = {r.raw["record_id"]: r for r in recs if r.raw["kernel"] == "vop_chain"}
    report = {}

    # VALU: dependent chain of one op per bundle -> latency = cycles per bundle
    latency = {}
    for op, mnem in (("add", "vadd.f32"), ("mul", "vmul.f32"), ("max", "vmax.f32")):
        r = vop[f"vop_chain__{op}_nv1_k1024"]
        n = Machine(r.program()).run(NoDma()).bundles_issued
        latency[mnem] = round(measured_cycles(r) / n)
    report["valu_latency"] = latency

    base_units = {}

    def with_units(units):
        return CoreParams(latency=dict(latency), units=units, issue_lead=LEAD)

    # EUP: latency from the nv1 exp chain, occupancy from nv8
    eup_recs = [vop["vop_chain__exp_nv1_k256"], vop["vop_chain__exp_nv8_k128"]]

    def eup_units(x):
        return {"eup": {"work": {"default": {"occupancy": 1, "latency": x[0], "results": 1}}, "pop_latency": 1}}

    # nv8 shows one exp per bundle with no stall, so occupancy is at most 1 cycle.
    eup, _ = fit(eup_recs[:1], lambda x: with_units(eup_units(x)), [5.0], ["latency"])
    base_units.update(eup_units([eup["latency"]]))
    report["eup"] = eup

    # XLU: permute (lane rotate) and xlane (lane reduction), 3 units each. Latency fitted on chains
    # 1, 8, 16 and 32 wide (phase3a, phase3f). All are latency-bound, so occupancy stays 1 (H8).
    xlu = {}
    wide = [r for r in load_records(f3_dir) if r.raw["kernel"] == "vop_chain"]
    for key, op in (("permute", "roll_lane"), ("xlane", "sum_lane")):
        rs = [r for r in list(vop.values()) + wide if r.raw["params"]["op"] == op]

        def xlu_units(x, key=key):
            return {"xlu": {"work": {key: {"occupancy": 1, "latency": x[0], "results": 1}}, "pop_latency": 1}}

        got, _ = fit(rs, lambda x: with_units(xlu_units(x)), [100.0], ["latency"])
        xlu[key] = {"occupancy": 1, "latency": got["latency"], "results": 1}
    base_units["xlu"] = {"work": xlu, "pop_latency": 1}
    report["xlu"] = xlu

    # MXU: bf16 vmatmul occupancy is 16 cycles (16 rows at one row/cycle; 1024->4096 rows step
    # measured 16.0 cycles per extra vmatmul). Fit the rest on reps=4, check on reps=16.
    mxu_recs = [r for r in recs if r.raw["kernel"] == "mxu_dot"]
    train = [r for r in mxu_recs if r.raw["params"]["reps"] == 4]
    check = [r for r in mxu_recs if r.raw["params"]["reps"] == 16]

    def mxu_units(x):
        u = dict(base_units)
        u["mxu"] = {
            "work": {"bf16": {"occupancy": 16, "latency": x[0], "interval": x[1], "results": 2}},
            "push_cycles": x[2],
            "load_latency": x[3],
            "pop_latency": 1,
        }
        return u

    mxu, res = fit(train, lambda x: with_units(mxu_units(x)), [120.0, 8.0, 8.0, 10.0],
                   ["result_latency", "result_interval", "push_cycles", "load_latency"])
    x = [mxu["result_latency"], mxu["result_interval"], mxu["push_cycles"], mxu["load_latency"]]
    final = with_units(mxu_units(x))
    report["mxu"] = mxu
    report["mxu_train_err"] = {r.raw["record_id"]: float(e) for r, e in zip(train, rel_errors(train, final))}
    report["mxu_check_err"] = {r.raw["record_id"]: float(e) for r, e in zip(check, rel_errors(check, final))}
    # f32 matmuls (phase3d): an f32 vreg is 8 rows, so vmatmul.f32 occupies the MXU 8 cycles
    # (1024 -> 4096 rows: 8.0 cycles per extra vmatmul per MXU) and yields one result vreg
    # (32 vmatmul.f32 : 32 vpop per MXU). Fitted: its result latency and the cost of an f32 weight
    # push (f32 weights take twice the vmatpush ops yet no more time); the rest stays as fitted on bf16.
    f32_recs = [r for r in load_records(f32_dir) if r.raw["kernel"] == "mxu_dot"]
    f32_train = [r for r in f32_recs if r.raw["params"]["reps"] == 4]
    f32_check = [r for r in f32_recs if r.raw["params"]["reps"] == 16]

    def with_f32(x):
        u = final.units | {"mxu": dict(final.units["mxu"])}
        u["mxu"]["work"] = dict(u["mxu"]["work"], f32={"occupancy": 8, "latency": x[0], "interval": 0, "results": 1})
        u["mxu"]["push_cycles_f32"] = x[1]
        return with_units(u)

    f32, _ = fit(f32_train, with_f32, [120.0, 2.0], ["result_latency", "push_cycles_f32"])
    final = with_f32([f32["result_latency"], f32["push_cycles_f32"]])
    report["mxu_f32"] = f32
    report["mxu_f32_train_err"] = {r.raw["record_id"]: float(e) for r, e in zip(f32_train, rel_errors(f32_train, final))}
    report["mxu_f32_check_err"] = {r.raw["record_id"]: float(e) for r, e in zip(f32_check, rel_errors(f32_check, final))}

    # Scalar load-to-use latency (phase3f sld_chain: j = table[j]); integer cycles, checked on the
    # dma_gather loops (phase3e), which use an sld result in the next bundle.
    from model.operands import integer_operands
    from model.predict import load_dma, simulate

    def cyc_with_ops(r, params, dma=None):
        prog = r.program()
        from model.predict import dma_model
        engine = dma_model(dma, CLOCK_GHZ) if dma else NoDma()
        return Machine(prog, core=Scoreboard(params), operands=integer_operands(r, prog)).run(engine).cycles

    sld_recs = [r for r in load_records(f3_dir) if r.raw["kernel"] == "sld_chain"]

    def with_sld(lat):
        return CoreParams(latency=dict(final.latency, sld=lat), units=final.units, issue_lead=LEAD)

    dma = load_dma()
    sld_err = {lat: [cyc_with_ops(r, with_sld(lat), dma) / measured_cycles(r) - 1 for r in sld_recs] for lat in range(1, 13)}
    sld_lat = min(sld_err, key=lambda lat: sum(e * e for e in sld_err[lat]))
    # Reported, not installed: an sld result latency of 7 and a 2-cycle address-register lag both fit
    # sld_chain exactly, and both make the calibration corpus worse (HARDWARE_FACTS F17, H11).
    report["sld_latency_if_result_latency"] = sld_lat
    report["sld_err"] = {r.raw["record_id"]: float(e) for r, e in zip(sld_recs, sld_err[sld_lat])}
    gather = [r for r in load_records(gather_dir) if r.raw["kernel"] == "dma_gather"]
    report["sld_check_dma_gather_err"] = {r.raw["record_id"]: float(cyc_with_ops(r, with_sld(sld_lat), dma) / measured_cycles(r) - 1) for r in gather}

    # Transposed bf16 weight pushes (phase3f mxu_dot rhs_t): one push cost, fit on reps=4, check reps=16.
    xt = [r for r in load_records(f3_dir) if r.raw["kernel"] == "mxu_dot"]
    xt_train = [r for r in xt if r.raw["params"]["reps"] == 4]
    xt_check = [r for r in xt if r.raw["params"]["reps"] == 16]

    def with_xpose(x):
        u = final.units | {"mxu": dict(final.units["mxu"], push_cycles_xpose=x[0])}
        return CoreParams(latency=final.latency, units=u, issue_lead=LEAD)

    xp, _ = fit(xt_train, with_xpose, [8.0], ["push_cycles_xpose"])
    final = with_xpose([xp["push_cycles_xpose"]])
    report["mxu_xpose"] = xp
    report["mxu_xpose_train_err"] = {r.raw["record_id"]: float(e) for r, e in zip(xt_train, rel_errors(xt_train, final))}
    report["mxu_xpose_check_err"] = {r.raw["record_id"]: float(e) for r, e in zip(xt_check, rel_errors(xt_check, final))}

    # XLU transposes: the unit accepts one input vreg every `occupancy` cycles and result j is ready
    # `latency` after input j (streaming). Fitted per form on phase3j xpose_chain (dependent chains)
    # and phase3n tgmm_tile (megablox tgmm's step: transposes beside weight pushes and matmuls).
    xc = [r for r in load_records(xpose_dir) if r.raw["kernel"] == "xpose_chain"]
    tiles = [r for r in load_records(tgmm_dir) if r.raw["kernel"] == "tgmm_tile"]

    def form_of(r):
        ms = {o.mnemonic for b in r.program().bundles for o in b.ops if o.mnemonic.startswith("vxpose")}
        return "b16bin" if any(".binary" in m for m in ms) else "b16" if any(".b16" in m for m in ms) else "b32"

    xpose = {}

    def with_xp(key, x):
        entry = {"occupancy": x[0], "latency": x[1], "interval": 0, "stream": True}
        u = final.units | {"xlu": dict(final.units["xlu"], xpose=dict(xpose, **{key: entry}))}
        return CoreParams(latency=final.latency, units=u, issue_lead=LEAD)

    xp_recs = {}
    for key in ("b32", "b16", "b16bin"):
        rs = [r for r in xc + tiles if form_of(r) == key]
        xp_recs[key] = rs
        got, _ = fit(rs, lambda x, key=key: with_xp(key, x), [8.0, 160.0], ["occupancy", "latency"])
        xpose[key] = {"occupancy": got["occupancy"], "latency": got["latency"], "interval": 0, "stream": True}
    final = CoreParams(latency=final.latency, units=final.units | {"xlu": dict(final.units["xlu"], xpose=xpose)}, issue_lead=LEAD)
    report["xlu_xpose"] = xpose
    allx = [r for rs in xp_recs.values() for r in rs]
    report["xlu_xpose_err"] = {r.raw["record_id"]: float(e) for r, e in zip(allx, rel_errors(allx, final))}

    # int8 matmuls (phase3p): one vmatmul.s8 streams 32 rows and yields 4 int32 result vregs. Fitted:
    # occupancy, result latency, result interval (plain s8 x s8), then the push cost of int8 weights,
    # transposed int8 weights and transposed f32 weights; fit on reps 4, check on reps 16.
    p3p = [r for r in load_records(int8_dir) if r.raw["kernel"] == "mxu_dot"]

    def kind_of(r):
        p = r.raw["params"]
        return ("s8" if p.get("lhs") == "s8" else "f32") + ("T" if p.get("rhs_t") else "")

    def with_mxu(work_s8=None, push=None):
        mx = dict(final.units["mxu"])
        if work_s8 is not None:
            mx["work"] = dict(mx["work"], s8={"occupancy": work_s8[0], "latency": work_s8[1], "interval": work_s8[2], "results": 4})
        if push:
            mx["push_by_kind"] = dict(mx.get("push_by_kind", {}), **push)
        return CoreParams(latency=final.latency, units=final.units | {"mxu": mx}, issue_lead=LEAD)

    s8 = [r for r in p3p if kind_of(r) == "s8"]
    got, _ = fit([r for r in s8 if r.raw["params"]["reps"] == 4], lambda x: with_mxu(work_s8=x), [16.0, 120.0, 8.0],
                 ["occupancy", "result_latency", "result_interval"])
    work_s8 = [got["occupancy"], got["result_latency"], got["result_interval"]]
    report["mxu_s8"] = got
    pushes = {}
    for kind, key, recs in (("s8", "s8", s8), ("s8T", "s8x", [r for r in p3p if kind_of(r) == "s8T"]),
                            ("f32T", "f32x", [r for r in p3p if kind_of(r) == "f32T"])):
        got, _ = fit([r for r in recs if r.raw["params"]["reps"] == 4],
                     lambda x, key=key: with_mxu(work_s8=work_s8, push=dict(pushes, **{key: x[0]})), [4.0], ["push_cycles"])
        pushes[key] = got["push_cycles"]
    final = with_mxu(work_s8=work_s8, push=pushes)
    report["mxu_push_by_kind"] = pushes
    report["mxu_s8_err"] = {r.raw["record_id"]: float(e) for r, e in zip(p3p, rel_errors(p3p, final))}

    # cast and roll_sublane chains were folded by the compiler (1 to 4 bundles); not usable.
    usable = [r for r in vop.values() if r.raw["params"]["op"] not in ("cast", "roll_sublane")]
    report["vop_err"] = {r.raw["record_id"]: float(e) for r, e in zip(usable, rel_errors(usable, final))}

    with open(out_json, "w") as f:
        json.dump({"core": {"latency": final.latency, "units": final.units, "issue_lead": LEAD}, "fit_report": report,
                   "provenance": {"sources": [batch_dir, f32_dir, f3_dir, gather_dir, xpose_dir, tgmm_dir, int8_dir], "script": "scripts/fit_core.py"}}, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:9])
