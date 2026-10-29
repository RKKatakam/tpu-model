"""Report on external test kernels (class external_*): error per kernel and ranking per sweep, ours vs
XLA's static estimate. These kernels are never used to fit the model.

usage: report_external.py OUT_MD BATCH_DIR [BATCH_DIR ...]
"""

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate import pct, row_for, spearman, sweep_key  # noqa: E402
from model.predict import load_records  # noqa: E402


def main(out_md, dirs):
    recs = [r for d in dirs for r in load_records(d) if r.raw["kernel_class"].startswith("external")]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(row_for, recs))
    for r, rec in zip(rows, recs):
        r["sweep"] = sweep_key(rec.raw)
        r["kernels"] = len(rec.hlo_names)
    md = ["# External test set (vllm-project/tpu-inference kernels)\n",
          "Never used to fit the model. Error = predicted / measured - 1 on median device time (sum over the "
          "program's Pallas kernels). XLA = static estimate where its profiler accepts the kernel.\n",
          "| kernel | n | ours median abs | ours worst | within 2% | within 5% | within 10% | XLA median abs (n) |",
          "|---|---|---|---|---|---|---|---|"]
    fams = defaultdict(list)
    for r in rows:
        fams[r["kernel"]].append(r)
    allr = []
    for k, rs in sorted(fams.items()):
        e = np.abs([r["ours"] / r["meas"] - 1 for r in rs if r["ours"] == r["ours"]])
        x = np.abs([r["xla"] / r["meas"] - 1 for r in rs if r["xla"] == r["xla"]])
        allr += list(e)
        md.append(f"| {k} | {len(rs)} | {pct(np.median(e)) if len(e) else '-'} | {pct(e.max()) if len(e) else '-'} | "
                  f"{np.sum(e <= .02)} | {np.sum(e <= .05)} | {np.sum(e <= .10)} | "
                  f"{pct(np.median(x)) + f' ({len(x)})' if len(x) else 'n/a'} |")
    a = np.array(allr)
    md.append(f"| **all** | {len(a)} | {pct(np.median(a))} | {pct(a.max())} | {np.sum(a <= .02)} | {np.sum(a <= .05)} | {np.sum(a <= .10)} | |\n")
    md += ["## Ranking within sweeps\n", "| sweep | n | Spearman ours | top-1 ours | regret ours |", "|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in rows:
        if r["ours"] == r["ours"]:
            groups[r["sweep"]].append(r)
    for key, g in sorted(groups.items()):
        if len(g) < 3:
            continue
        meas, ours = [r["meas"] for r in g], [r["ours"] for r in g]
        best, bo = int(np.argmin(meas)), int(np.argmin(ours))
        md.append(f"| {key[:80]} | {len(g)} | {spearman(ours, meas):.2f} | {'yes' if bo == best else 'no'} | {pct(meas[bo] / meas[best] - 1)} |")
    md += ["\n## All records\n", "| id | kernels | measured us | ours | XLA |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["id"]):
        md.append(f"| {r['id']} | {r['kernels']} | {r['meas'] / 1e3:.2f} | {r['ours'] / r['meas'] - 1:+.1%} | {r['xla'] / r['meas'] - 1:+.1%} |")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(out_md.replace(".md", ".json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n".join(md[: md.index("\n## All records\n")]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
