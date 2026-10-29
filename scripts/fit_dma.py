"""Fit the shared-HBM DMA model.

usage: fit_dma.py PHASE1_DIR PHASE3B_DIR OUT_JSON
Read curve: bench dma_seq (phase 1). Write curve: bench dma_out (phase 3b). Both per-copy
= (t(8 copies) - t(1 copy)) / 7. Aggregate read bandwidth, mixed read/write bandwidth, and
per-DMA issue cost are fitted on bench dma_conc and dma_inout only.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dma import FifoHbm, SharedHbm  # noqa: E402
from model.machine import Machine  # noqa: E402
from model.predict import load_records  # noqa: E402

CLOCK_GHZ = 1.5  # HARDWARE_FACTS F7


def per_copy_table(recs, kernel, size_of):
    by = {r.raw["record_id"]: r for r in recs if r.raw["kernel"] == kernel}
    out = []
    for rid, r in by.items():
        if r.raw["params"]["reps"] != 8:
            continue
        one = by[rid.replace("_x8", "_x1")]
        out.append([size_of(r.raw["params"]), (r.measured_ns() - one.measured_ns()) / 7])
    return sorted(out)


def marginal_gbps(table, lo=1 << 20):
    """Bandwidth of the large-size end of a single-stream curve (bytes per ns = GB/s)."""
    (b0, t0), (b1, t1) = [p for p in table if p[0] == lo][0], table[-1]
    return (b1 - b0) / (t1 - t0)


BW_WRITE = None
ENGINE = {"shared": SharedHbm, "fifo": FifoHbm}[os.environ.get("TPUM_DMA_ENGINE", "fifo")]


def model(read_ns, write_ns, x):
    bw_read, bw_mixed, issue_ns, mix_ns = x
    to_cyc = lambda tbl: [(b, ns * CLOCK_GHZ) for b, ns in tbl]  # noqa: E731
    return ENGINE(to_cyc(read_ns), to_cyc(write_ns), bw_read / CLOCK_GHZ, BW_WRITE / CLOCK_GHZ,
                     bw_mixed / CLOCK_GHZ, issue_ns * CLOCK_GHZ, mix_ns * CLOCK_GHZ)


def errors(recs, read_ns, write_ns, x):
    out = []
    for r in recs:
        cyc = Machine(r.program()).run(model(read_ns, write_ns, x)).cycles
        out.append(cyc / (r.measured_ns() * CLOCK_GHZ) - 1)
    return np.array(out)


def main(p1, p3b, out_json):
    r1, r3 = load_records(p1), load_records(p3b)
    read_ns = per_copy_table(r1, "dma_seq", lambda p: p["rows"] * 512)
    write_ns = per_copy_table(r3, "dma_out", lambda p: p["rows"] * 512)
    fit_recs = [r for r in r3 if r.raw["kernel"] in ("dma_conc", "dma_inout")]
    global BW_WRITE
    # No concurrent-write microbenchmark yet: write aggregate = large-size slope of the write curve.
    BW_WRITE = marginal_gbps(write_ns)

    loss = lambda x: float(np.sum(errors(fit_recs, read_ns, write_ns, x) ** 2))  # noqa: E731
    res = minimize(loss, [730.0, 600.0, 12.0, 50.0], method="Nelder-Mead", options={"xatol": 0.5, "fatol": 1e-8, "maxiter": 300})
    x = res.x
    errs = errors(fit_recs, read_ns, write_ns, x)
    check = [r for r in r1 + r3 if r.raw["kernel"] in ("dma_seq", "dma_out")]
    check_errs = errors(check, read_ns, write_ns, x)
    out = {
        "dma": {"read_table_ns": read_ns, "write_table_ns": write_ns, "bw_read_GBps": x[0],
                "bw_write_GBps": BW_WRITE, "bw_mixed_GBps": x[1], "issue_ns": x[2], "mix_ns": x[3]},
        "fit_report": {
            "fit_err": {r.raw["record_id"]: float(e) for r, e in zip(fit_recs, errs)},
            "single_stream_err": {r.raw["record_id"]: float(e) for r, e in zip(check, check_errs)},
        },
        "provenance": {"sources": [p1, p3b], "script": "scripts/fit_dma.py", "engine": ENGINE.__name__},
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["dma"] | {"max_fit_err": float(np.abs(errs).max()), "max_single_err": float(np.abs(check_errs).max())}, indent=1))
    for r, e in sorted(zip(fit_recs, errs), key=lambda t: -abs(t[1]))[:10]:
        print(f"  {r.raw['record_id']:28s} {e:+.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
