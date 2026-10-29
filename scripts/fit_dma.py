"""Fit the DMA model (DirFifoHbm) on DMA microbenchmarks only.

usage: fit_dma.py PHASE1_DIR PHASE3B_DIR PHASE3C_DIR OUT_JSON
Read curve: bench dma_seq (phase 1). Write curve: bench dma_out (phase 3b). Both per-copy
= (t(8 copies) - t(1 copy)) / 7. Read and write bandwidth, read/write turnaround and per-DMA
issue cost are fitted on bench dma_stream (pipelined read/write mixes), dma_conc (concurrent
reads) and dma_inout (read then write); dma_seq / dma_out / dma_strided are reported as checks.
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
FIT = ("dma_stream", "dma_conc", "dma_inout")
CHECK = ("dma_seq", "dma_out", "dma_strided")


def per_copy_table(recs, kernel, size_of):
    by = {r.raw["record_id"]: r for r in recs if r.raw["kernel"] == kernel}
    out = []
    for rid, r in by.items():
        if r.raw["params"]["reps"] != 8:
            continue
        one = by[rid.replace("_x8", "_x1")]
        out.append([size_of(r.raw["params"]), (r.measured_ns() - one.measured_ns()) / 7])
    return sorted(out)


def params_of(read_ns, write_ns, x):
    bw_read, bw_write, turnaround_ps, issue_ns = x
    return {"engine": "DirFifoHbm", "read_table_ns": read_ns, "write_table_ns": write_ns,
            "bw_read_GBps": bw_read, "bw_write_GBps": bw_write,
            "turnaround_ns_per_byte": turnaround_ps / 1000, "issue_ns": issue_ns}


def errors(recs, dma):
    out = []
    for r in recs:
        cyc = Machine(r.program()).run(dma_model(dma, CLOCK_GHZ)).cycles
        out.append(cyc / (r.measured_ns() * CLOCK_GHZ) - 1)
    return np.array(out)


def main(p1, p3b, p3c, out_json):
    r1, r3, r3c = load_records(p1), load_records(p3b), load_records(p3c)
    read_ns = per_copy_table(r1, "dma_seq", lambda p: p["rows"] * 512)
    write_ns = per_copy_table(r3, "dma_out", lambda p: p["rows"] * 512)
    everything = r1 + r3 + r3c
    fit_recs = [r for r in everything if r.raw["kernel"] in FIT]
    check = [r for r in everything if r.raw["kernel"] in CHECK]

    # x = (bw_read GB/s, bw_write GB/s, turnaround ps/byte, issue ns); start from the dma_stream
    # closed form: 751 / 655 GB/s sustained, 0.2 ps/byte overlapped.
    loss = lambda x: float(np.sum(errors(fit_recs, params_of(read_ns, write_ns, x)) ** 2))  # noqa: E731
    res = minimize(loss, [751.0, 655.0, 0.2, 11.4], method="Nelder-Mead",
                   options={"xatol": 0.05, "fatol": 1e-8, "maxiter": 400})
    dma = params_of(read_ns, write_ns, res.x)
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
    main(*sys.argv[1:5])
