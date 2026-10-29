"""Validation report: per-class error and autotune ranking, ours vs XLA's static estimate.

usage: validate.py OUT_MD BATCH_DIR [BATCH_DIR ...]
Uses model/params/{phase1,core,dma}.json. Splits come from corpus/split.py (fixed in advance).
"""

import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.split import split_of  # noqa: E402
from model.predict import Params, load_core, load_dma, load_records, predict_record_ns  # noqa: E402

TILE_KEYS = {"m_blocks", "d_blocks", "block_q", "block_k_major", "block_k", "block_kv", "block_kv_compute", "pages_per_compute_block",
             "tm", "tk", "tn", "block_rows", "block_cols", "bm", "bn", "bk"}
CLASSES = ("compute", "streaming", "latency")
# User directive (2026-10-29): splash is the TPU attention benchmark; flash is reported separately.
SUPPLEMENTARY = {"flash"}


def sweep_key(rec):
    p = {k: v for k, v in rec["params"].items() if k not in TILE_KEYS}
    return rec["kernel"] + json.dumps(p, sort_keys=True)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pct(x):
    return f"{100 * x:.1f}%"


def stats(errs):
    a = np.abs(np.array(errs))
    if not len(a):
        return "-", "-", "-"
    return pct(np.median(a)), pct(np.percentile(a, 90)), pct(a.max())


def row_for(r):
    """Replay one record; a replay failure is reported (stderr) and scored NaN, never hidden."""
    P = Params.load("model/params/phase1.json")
    try:
        ours, traces = predict_record_ns(r, P, load_core(), load_dma())
        for tr in traces:
            if tr.unmodeled_smem_dmas or tr.halts:
                print(f"warning {r.raw['record_id']}: {tr.unmodeled_smem_dmas} unmodeled SMEM DMAs, "
                      f"halts at {tr.halts[:3]}", file=sys.stderr)
    except Exception as e:
        print(f"replay failed {r.raw['record_id']}: {e!r}", file=sys.stderr)
        ours = float("nan")
    est = r.xla_estimated_cycles()
    return {
        "id": r.raw["record_id"], "kernel": r.raw["kernel"], "class": r.raw["kernel_class"],
        "split": split_of(r.raw["record_id"]), "sweep": sweep_key(r.raw),
        "meas": r.measured_ns(), "ours": ours, "xla": est / P.clock_ghz if est else float("nan"),
    }


def main(out_md, dirs):
    recs = [r for d in dirs for r in load_records(d) if r.raw["kernel_class"] in CLASSES]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(row_for, recs))
    md = []
    w = md.append
    w("# Validation\n")
    w("Error = predicted / measured - 1 on the median device time. XLA = `estimated_cycles` from "
      "`--xla_jf_dump_static_llo_profile_to` at 1.5 GHz.\n")
    headline = [r for r in rows if r["kernel"] not in SUPPLEMENTARY]
    tables = [(f"split = {split}", [r for r in headline if r["split"] == split]) for split in ("valid", "calib")]
    tables.append(("supplementary kernels (" + ", ".join(sorted(SUPPLEMENTARY)) + "), all splits",
                   [r for r in rows if r["kernel"] in SUPPLEMENTARY]))
    for title, subset in tables:
        w(f"## Per-class error, {title}\n")
        w("| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for c in CLASSES:
            sel = [r for r in subset if r["class"] == c and not np.isnan(r["ours"])]
            eo = [r["ours"] / r["meas"] - 1 for r in sel]
            ex = [r["xla"] / r["meas"] - 1 for r in sel if not np.isnan(r["xla"])]
            w5 = sum(abs(e) <= 0.05 for e in eo)
            w10 = sum(abs(e) <= 0.10 for e in eo)
            w(f"| {c} | {len(sel)} | {' | '.join(stats(eo))} | {' | '.join(stats(ex))} | {w5}/{len(sel)} | {w10}/{len(sel)} |")
        w("")
    w("## Ranking within autotune sweeps (all splits)\n")
    w("| sweep | n | Spearman ours | Spearman XLA | top-1 ours | top-1 XLA | top-3 overlap ours | regret ours | regret XLA |")
    w("|---|---|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in rows:
        groups[r["sweep"]].append(r)
    for key, g in sorted(groups.items()):
        if len(g) < 3:
            continue
        meas = [r["meas"] for r in g]
        ours = [r["ours"] for r in g]
        xla = [r["xla"] for r in g]
        best = int(np.argmin(meas))
        bo = int(np.argmin(ours))
        top3 = set(np.argsort(meas)[:3])
        if all(np.isnan(xla)):  # XLA's static profiler rejects the kernel (e.g. splash)
            sx = t1x = rx = "n/a"
        else:
            bx = int(np.nanargmin(xla))
            sx, t1x, rx = f"{spearman(xla, meas):.2f}", "yes" if bx == best else "no", pct(meas[bx] / meas[best] - 1)
        w(f"| {key[:70]} | {len(g)} | {spearman(ours, meas):.2f} | {sx} | "
          f"{'yes' if bo == best else 'no'} | {t1x} | "
          f"{len(top3 & set(np.argsort(ours)[:3]))}/3 | {pct(meas[bo] / meas[best] - 1)} | {rx} |")
    w("\n## All records\n")
    w("| id | class | split | measured µs | ours | XLA |")
    w("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["id"]):
        w(f"| {r['id']} | {r['class']} | {r['split']} | {r['meas'] / 1e3:.2f} | {r['ours'] / r['meas'] - 1:+.1%} | "
          f"{r['xla'] / r['meas'] - 1:+.1%} |")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(out_md.replace(".md", ".json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n".join(md[: md.index("## All records\n")] if "## All records\n" in md else md))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
