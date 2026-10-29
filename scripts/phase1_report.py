"""Phase 1 feasibility: clock fit (1a), DMA size sweep (1b), matmul tile prediction (1c).

usage: phase1_report.py DEVICE_BATCH_DIR OUT_DIR
Writes OUT_DIR/phase1.md, OUT_DIR/phase1_*.png and model/params/phase1.json.
"""

import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.machine import Machine  # noqa: E402
from model.predict import Params, load_records, predict_ns, simulate  # noqa: E402

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


class NoDma:
    def complete(self, d, inflight, t):
        raise AssertionError("unexpected DMA in a no-DMA microbenchmark")


def issued(r):
    return Machine(r.program()).run(NoDma()).bundles_issued


def fit_line(x, y):
    """Least squares y = a + b x; returns a, b, residuals."""
    A = np.vstack([np.ones(len(x)), x]).T
    (a, b), *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return a, b, np.asarray(y) - (a + b * np.asarray(x))


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color="#e4e3df", linewidth=0.6)
    for s in ax.spines.values():
        s.set_color("#d0cfca")


def main(batch_dir, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    recs = load_records(batch_dir)
    by = {r.raw["record_id"]: r for r in recs}
    md = []
    w = md.append

    # 1a: clock fit on hazard-free code. nv=8 rounds touch each vreg every 2 bundles, so no op
    # waits on its predecessor; nv=1 is a back-to-back dependent chain and is reported separately.
    straight = [r for r in recs if r.raw["kernel"] == "valu_straight"]
    x = np.array([issued(r) for r in straight])
    y = np.array([r.measured_ns() for r in straight])
    free = [i for i, r in enumerate(straight) if r.raw["params"]["nv"] == 8]
    a, slope, res = fit_line(x[free], y[free])
    f_ghz = 1 / slope
    w("## 1a. Clock from straight-line kernels\n")
    w(f"Fit over {len(free)} hazard-free kernels (nv=8 independent vadd chains, no loops, no DMA): time = a + bundles / f.\n")
    w(f"- f = **{f_ghz:.4f} GHz**, a = **{a:.2f} ns** fixed overhead")
    w(f"- residuals: max |r| = {np.abs(res).max():.2f} ns\n")
    w("| record | bundles | measured ns | a + bundles/f | measured cycles / bundle |")
    w("|---|---|---|---|---|")
    for r, xi, yi in sorted(zip(straight, x, y), key=lambda t: (t[0].raw["params"]["nv"], t[1])):
        w(f"| {r.raw['record_id']} | {xi} | {yi:.1f} | {a + xi / f_ghz:.1f} | {yi * f_ghz / xi:.3f} |")
    dep = [i for i, r in enumerate(straight) if r.raw["params"]["nv"] == 1 and r.raw["params"]["k"] > 0]
    da, db, _ = fit_line(x[dep], y[dep])
    w(f"\nnv=1 (each vadd.f32 reads the previous bundle's result): {db * f_ghz:.3f} cycles per bundle. "
      "The dump has no padding between them, so the hardware interlocks: **vadd.f32 result latency = 2 cycles**.")
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    for i, nv in enumerate((8, 1)):
        sel = [j for j, r in enumerate(straight) if r.raw["params"]["nv"] == nv]
        ax.plot(x[sel], y[sel] / 1e3, "o", ms=6, color=SERIES[i], label=f"nv={nv}")
    xs = np.linspace(0, x.max(), 50)
    ax.plot(xs, (a + xs / f_ghz) / 1e3, "-", lw=2, color=MUTED, label=f"1 bundle/cycle at {f_ghz:.3f} GHz")
    style(ax, "Straight-line kernels: device time vs bundles", "bundles issued", "device time (µs)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase1_clock.png"), dpi=130)

    # loop overhead check: predicted from 1a vs measured
    w("\n## Loops: does a taken branch cost more than its bundles?\n")
    w("| record | bundles issued | measured ns | a + bundles/f | extra ns per iteration |")
    w("|---|---|---|---|---|")
    for r in sorted((r for r in recs if r.raw["kernel"] == "valu_loop"), key=lambda r: (r.raw["params"]["k"], r.raw["params"]["trip"])):
        n = issued(r)
        pred = a + n / f_ghz
        trip = r.raw["params"]["trip"]
        w(f"| {r.raw['record_id']} | {n} | {r.measured_ns():.1f} | {pred:.1f} | {(r.measured_ns() - pred) / trip:+.3f} |")

    # 1b: DMA size sweep
    w("\n## 1b. Single-stream HBM->VMEM DMA vs size\n")
    w("Per-copy time = (t(8 copies) - t(1 copy)) / 7, minus 3 bundles of issue/wait/syncadd at f.\n")
    w("| bytes | t(1) ns | t(8) ns | per-copy ns | GB/s |")
    w("|---|---|---|---|---|")
    sizes, per = [], []
    for rows in sorted({r.raw["params"]["rows"] for r in recs if r.raw["kernel"] == "dma_seq"}):
        t1 = by[f"dma_seq__r{rows}_x1"].measured_ns()
        t8 = by[f"dma_seq__r{rows}_x8"].measured_ns()
        pc = (t8 - t1) / 7 - 3 / f_ghz
        nbytes = rows * 128 * 4
        sizes.append(nbytes)
        per.append(pc)
        w(f"| {nbytes} | {t1:.1f} | {t8:.1f} | {pc:.1f} | {nbytes / pc:.1f} |")
    sizes, per = np.array(sizes), np.array(per)
    big = sizes >= 64 * 1024
    lat_ns, inv_bw, res_b = fit_line(sizes[big], per[big])
    bw = 1 / inv_bw
    w(f"\nFit on sizes >= 64 KiB: per-copy = L + bytes / B with L = **{lat_ns:.1f} ns**, B = **{bw:.1f} GB/s** "
      f"(max |residual| {np.abs(res_b).max():.1f} ns).")
    small = per[sizes <= 4096]
    w(f"Copies <= 4 KiB take {small.min():.1f} to {small.max():.1f} ns each.")

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    ax.loglog(sizes, per, "o", ms=6, color=SERIES[0], label="measured per-copy time")
    xs = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 60)
    ax.loglog(xs, lat_ns + xs / bw, "-", lw=2, color=MUTED, label=f"L={lat_ns:.0f} ns + bytes/{bw:.0f} GB/s")
    style(ax, "HBM->VMEM DMA, one stream, start then wait", "bytes per copy", "time per copy (ns)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase1_dma.png"), dpi=130)

    params = Params(clock_ghz=f_ghz, overhead_ns=a, dma_table_ns=[[int(b), float(t)] for b, t in zip(sizes, per)])
    os.makedirs("model/params", exist_ok=True)
    params.dump(
        "model/params/phase1.json",
        {"source": batch_dir, "fit": "scripts/phase1_report.py", "clock": "1a straight-line", "dma": "1b per-copy table, log-size interpolation"},
    )

    # 1c: matmul tile configs
    w("\n## 1c. Double-buffered matmul 2048^3 bf16, prediction vs measured\n")
    w("Parameters come only from 1a (clock, overhead) and 1b (per-copy DMA time table). "
      "Vector/MXU ops cost one issue cycle each; interlock stalls are not modelled yet. "
      "XLA's estimate is converted with the fitted clock.\n")
    w("| tiles (bm,bn,bk) | measured µs | ours µs | ours err | XLA est µs | XLA err | bundles issued | DMA stall cycles |")
    w("|---|---|---|---|---|---|---|---|")
    errs = []
    for r in sorted((r for r in recs if r.raw["kernel"] == "matmul"), key=lambda r: r.raw["record_id"]):
        tr = simulate(r.program(), params)
        ours = predict_ns(tr, params)
        meas = r.measured_ns()
        xla = r.xla_estimated_cycles() / f_ghz
        p = r.raw["params"]
        errs.append(ours / meas - 1)
        w(f"| {p['bm']},{p['bn']},{p['bk']} | {meas / 1e3:.2f} | {ours / 1e3:.2f} | {ours / meas - 1:+.1%} | "
          f"{xla / 1e3:.2f} | {xla / meas - 1:+.1%} | {tr.bundles_issued} | {tr.stall_cycles:.0f} |")
    w(f"\nGate 1 (within ~10% on every config): **{'PASS' if max(abs(e) for e in errs) <= 0.10 else 'FAIL'}** "
      f"(worst {max(errs, key=abs):+.1%}).")

    with open(os.path.join(out_dir, "phase1.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
