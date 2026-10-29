"""Fit vector-side core parameters (latencies, EUP/XLU/MXU unit timings) from phase3a microbenchmarks.

usage: fit_core.py PHASE3A_DIR PHASE3D_DIR OUT_JSON
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



def cycles(record, params):
    return Machine(record.program(), core=Scoreboard(params)).run(NoDma()).cycles


def measured_cycles(record):
    return record.measured_ns() * CLOCK_GHZ


def rel_errors(records, params):
    return np.array([cycles(r, params) / measured_cycles(r) - 1 for r in records])


def fit(objective_records, build, x0, names):
    def loss(x):
        return float(np.sum(rel_errors(objective_records, build(x)) ** 2))

    res = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 0.25, "fatol": 1e-7, "maxiter": 400})
    return dict(zip(names, res.x)), res


def main(batch_dir, f32_dir, out_json):
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
        return CoreParams(latency=dict(latency), units=units)

    # EUP: latency from the nv1 exp chain, occupancy from nv8
    eup_recs = [vop["vop_chain__exp_nv1_k256"], vop["vop_chain__exp_nv8_k128"]]

    def eup_units(x):
        return {"eup": {"work": {"default": {"occupancy": 1, "latency": x[0], "results": 1}}, "pop_latency": 1}}

    # nv8 shows one exp per bundle with no stall, so occupancy is at most 1 cycle.
    eup, _ = fit(eup_recs[:1], lambda x: with_units(eup_units(x)), [5.0], ["latency"])
    base_units.update(eup_units([eup["latency"]]))
    report["eup"] = eup

    # XLU: permute (lane rotate) and xlane (lane reduction), 3 units each. Latency from the nv1
    # chain. The nv8 run is still latency-bound (8 chains x ~105 cycles), so it cannot identify
    # occupancy; occupancy stays 1 until a wider microbenchmark measures it.
    xlu = {}
    for key, op in (("permute", "roll_lane"), ("xlane", "sum_lane")):
        rs = [vop[f"vop_chain__{op}_nv1_k256"]]

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

    # cast and roll_sublane chains were folded by the compiler (1 to 4 bundles); not usable.
    usable = [r for r in vop.values() if r.raw["params"]["op"] not in ("cast", "roll_sublane")]
    report["vop_err"] = {r.raw["record_id"]: float(e) for r, e in zip(usable, rel_errors(usable, final))}

    with open(out_json, "w") as f:
        json.dump({"core": {"latency": final.latency, "units": final.units}, "fit_report": report,
                   "provenance": {"sources": [batch_dir, f32_dir], "script": "scripts/fit_core.py"}}, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
