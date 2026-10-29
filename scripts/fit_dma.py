"""Fit the DMA model (DirFifoHbm) on DMA microbenchmarks only.

usage: fit_dma.py PHASE1_DIR PHASE3B_DIR PHASE3C_DIR OUT_JSON
Read curve: bench dma_seq (phase 1). Write curve: bench dma_out (phase 3b). Both per-copy
= (t(8 copies) - t(1 copy)) / 7. Sustained read and write bandwidth
and the read/write overlap cost come in closed form from the 128- and 512-row dma_stream runs
(HARDWARE_FACTS F14: time = R/bw_read + W/bw_write + turnaround * min(R, W)). Per-DMA issue cost
and mixed-direction start-up are then fitted on dma_stream, dma_conc, dma_inout, dma_seq and
dma_out; dma_strided is reported as a check.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.machine import Machine  # noqa: E402
from model.predict import dma_model, load_records  # noqa: E402

CLOCK_GHZ = 1.5  # HARDWARE_FACTS F7
FIT = ("dma_stream", "dma_conc", "dma_inout", "dma_seq", "dma_out")
CHECK = ("dma_strided",)


def per_copy_table(recs, kernel, size_of):
    by = {r.raw["record_id"]: r for r in recs if r.raw["kernel"] == kernel}
    out = []
    for rid, r in by.items():
        if r.raw["params"]["reps"] != 8:
            continue
        one = by[rid.replace("_x8", "_x1")]
        out.append([size_of(r.raw["params"]), (r.measured_ns() - one.measured_ns()) / 7])
    return sorted(out)


def params_of(read_ns, write_ns, sustained, x):
    issue_ns, mix_ns = x
    return {"engine": "DirFifoHbm", "read_table_ns": read_ns, "write_table_ns": write_ns, **sustained,
            "issue_ns": issue_ns, "mix_ns": mix_ns}


def _cycles(args):
    r, dma = args
    return Machine(r.program()).run(dma_model(dma, CLOCK_GHZ)).cycles / (r.measured_ns() * CLOCK_GHZ) - 1


POOL = None


def errors(recs, dma):
    return np.array(list(POOL.map(_cycles, [(r, dma) for r in recs])))


def sustained_fit(stream_recs):
    """Bandwidths (GB/s) and overlap cost (ns/byte) from the bandwidth-bound stream runs."""
    rows = []
    for r in stream_recs:
        p = r.raw["params"]
        if p["block_rows"] < 128:
            continue  # 32-row blocks are latency-bound
        unit = p["rows"] * p["cols"] * 4
        R, W = unit * p["reads"], unit * p["writes"]
        rows.append(([R, W, min(R, W)], r.measured_ns()))
    a = np.array([x for x, _ in rows], dtype=float)
    t = np.array([y for _, y in rows])
    # relative least squares: divide each equation by its measured time
    coef, *_ = np.linalg.lstsq(a / t[:, None], np.ones(len(t)), rcond=None)
    inv_r, inv_w, turn = coef
    return {"bw_read_GBps": 1 / inv_r, "bw_write_GBps": 1 / inv_w, "turnaround_ns_per_byte": turn}


def main(p1, p3b, p3c, out_json):
    r1, r3, r3c = load_records(p1), load_records(p3b), load_records(p3c)
    read_ns = per_copy_table(r1, "dma_seq", lambda p: p["rows"] * 512)
    write_ns = per_copy_table(r3, "dma_out", lambda p: p["rows"] * 512)
    everything = r1 + r3 + r3c
    fit_recs = [r for r in everything if r.raw["kernel"] in FIT]
    check = [r for r in everything if r.raw["kernel"] in CHECK]

    sustained = sustained_fit([r for r in r3c if r.raw["kernel"] == "dma_stream"])
    loss = lambda x: float(np.sum(errors(fit_recs, params_of(read_ns, write_ns, sustained, x)) ** 2))  # noqa: E731
    res = minimize(loss, [12.0, 80.0], method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-8, "maxiter": 300})
    dma = params_of(read_ns, write_ns, sustained, res.x)
    fit_err, check_err = errors(fit_recs, dma), errors(check, dma)
    out = {
        "dma": dma,
        "fit_report": {
            "fit_err": {r.raw["record_id"]: float(e) for r, e in zip(fit_recs, fit_err)},
            "check_err": {r.raw["record_id"]: float(e) for r, e in zip(check, check_err)},
        },
        "provenance": {"sources": [p1, p3b, p3c], "script": "scripts/fit_dma.py", "engine": "DirFifoHbm"},
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    shown = {k: v for k, v in dma.items() if not k.endswith("table_ns")}
    print(json.dumps(shown | {"max_fit_err": float(np.abs(fit_err).max()),
                              "max_check_err": float(np.abs(check_err).max())}, indent=1))
    for name, recs, errs in (("fit", fit_recs, fit_err), ("check", check, check_err)):
        print(f"{name}: median |err| {np.median(np.abs(errs)):.1%}, within 5%: {int(np.sum(np.abs(errs) <= 0.05))}/{len(errs)}")
        for r, e in sorted(zip(recs, errs), key=lambda t: -abs(t[1]))[:8]:
            print(f"  {r.raw['record_id']:30s} {e:+.3f}")


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as POOL:
        main(*sys.argv[1:5])
