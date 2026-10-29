"""Tiling search with the cost model: compile every candidate without a TPU, rank them by predicted
time, then time only the best few on the TPU.

usage:
  search.py plan SPACE          write batches/search_SPACE.json (deviceless AOT compiles, CPU runtime)
  search.py rank SPACE [K]      predict every compiled candidate (data/raw/search_SPACE), write
                                reports/search_SPACE.md and batches/search_SPACE_verify.json: the top K
                                plus controls spread over the predicted ranking
  search.py check SPACE         compare predictions with the verify batch's measurements
                                (data/raw/search_SPACE_verify): top-1 regret, rank correlation
"""

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.predict import Params, load_core, load_dma, load_records, predict_record_ns  # noqa: E402

# space -> (kernel, kernel_tag, fixed params, {tunable param: values}, feasibility check)
SPACES = {
    "matmul_tpu_4k": ("matmul_tpu", "matmul", {"m": 4096, "k": 4096, "n": 4096},
                      {"bm": [128, 256, 512, 1024], "bn": [128, 256, 512, 1024], "bk": [128, 256, 512, 1024]},
                      lambda p: p["bm"] * p["bk"] + p["bk"] * p["bn"] <= 2 * 1024 * 1024),
    "splash_s4096": ("splash", "splash_mha_fwd", {"h": 8, "s": 4096, "d": 128, "mask": "causal"},
                     {"block_q": [128, 256, 512, 1024], "block_kv": [128, 256, 512, 1024, 2048],
                      "block_kv_compute": [128, 256, 512, 1024]},
                     lambda p: p["block_kv_compute"] <= p["block_kv"] and p["block_kv"] % p["block_kv_compute"] == 0),
    "tgmm_8k": ("tgmm", "tgmm", {"m": 8192, "k": 1024, "n": 1024, "groups": 8},
                {"tm": [128, 256, 512, 1024], "tk": [128, 256, 512], "tn": [128, 256, 512, 1024]},
                lambda p: True),
}


def candidates(space):
    kernel, tag, fixed, grid, ok = SPACES[space]
    for values in itertools.product(*grid.values()):
        p = fixed | dict(zip(grid, values))
        if ok(p):
            yield kernel, tag, p


def rid(space, p):
    _, _, fixed, grid, _ = SPACES[space]
    return f"{space}__" + "_".join(f"{k}{p[k]}" for k in grid)


def flags():
    with open(os.path.join(ROOT, "batches", "corpus3_part0.json")) as f:
        return json.load(f)["flags"]


def plan(space):
    cfgs = [{"record_id": rid(space, p), "kernel": k, "kernel_tag": tag, "params": p, "mode": "aot",
             "topology": "v5e:1x1", "seed": 0, "timeout_s": 600} for k, tag, p in candidates(space)]
    out = os.path.join(ROOT, "batches", f"search_{space}.json")
    with open(out, "w") as f:
        json.dump({"batch_id": f"search_{space}", "flags": flags(), "configs": cfgs}, f, indent=1)
    print(out, len(cfgs), "candidates")


def _predict(r):
    try:
        ns, _ = predict_record_ns(r, Params.load(os.path.join(ROOT, "model/params/phase1.json")),
                                  load_core(os.path.join(ROOT, "model/params/core.json")),
                                  load_dma(os.path.join(ROOT, "model/params/dma.json")))
        return r.raw["record_id"], ns, None
    except Exception as e:  # a candidate the replay cannot handle is reported, not ranked
        return r.raw["record_id"], float("nan"), repr(e)[:200]


def predictions(batch_dir):
    recs = load_records(batch_dir)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        return recs, list(pool.map(_predict, recs))


def rank(space, k=6):
    recs, preds = predictions(os.path.join(ROOT, "data", "raw", f"search_{space}"))
    by_id = {r.raw["record_id"]: r for r in recs}
    ok = sorted((p for p in preds if p[1] == p[1]), key=lambda p: p[1])
    failed = [p for p in preds if p[1] != p[1]]
    md = [f"# Search {space}\n", f"{len(preds)} compiled candidates, {len(failed)} not replayable. "
          "Predicted device time from the cost model (no TPU timing).\n",
          "| rank | candidate | predicted us |", "|---|---|---|"]
    md += [f"| {i + 1} | {r} | {ns / 1e3:.2f} |" for i, (r, ns, _) in enumerate(ok)]
    md += ["\n## Not replayable\n"] + [f"- {r}: {e}" for r, _, e in failed]
    with open(os.path.join(ROOT, "reports", f"search_{space}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    # verify: the top k, plus controls at evenly spaced predicted ranks (for rank correlation)
    picks = [r for r, _, _ in ok[:k]]
    picks += [ok[int(i)][0] for i in np.linspace(k, len(ok) - 1, 6) if ok[int(i)][0] not in picks]
    verify = []
    for r in picks:
        c = dict(by_id[r].raw["config"], mode="device", reps=50, warmup=5)
        c.pop("topology", None)
        verify.append(c)
    out = os.path.join(ROOT, "batches", f"search_{space}_verify.json")
    with open(out, "w") as f:
        json.dump({"batch_id": f"search_{space}_verify", "flags": flags(), "configs": verify}, f, indent=1)
    print("\n".join(md[:k + 4]))
    print(out, len(verify), "to time")


def check(space):
    recs, preds = predictions(os.path.join(ROOT, "data", "raw", f"search_{space}_verify"))
    rows = sorted(((r, ns, rec.measured_ns()) for rec, (r, ns, _) in zip(recs, preds) if ns == ns), key=lambda x: x[1])
    meas = np.array([m for _, _, m in rows])
    pred = np.array([p for _, p, _ in rows])
    best = meas.argmin()
    rho = np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(meas)))[0, 1]
    print(f"| candidate | predicted us | measured us | error |\n|---|---|---|---|")
    for r, p, m in rows:
        print(f"| {r} | {p / 1e3:.2f} | {m / 1e3:.2f} | {p / m - 1:+.1%} |")
    print(f"\nmodel's pick {rows[0][0]}: {meas[0] / 1e3:.2f} us; best measured {rows[best][0]}: {meas[best] / 1e3:.2f} us; "
          f"regret {meas[0] / meas[best] - 1:+.1%}; Spearman {rho:.2f}; median |error| {np.median(np.abs(pred / meas - 1)):.1%}")


if __name__ == "__main__":
    cmd, space = sys.argv[1], sys.argv[2]
    {"plan": lambda: plan(space), "rank": lambda: rank(space, int(sys.argv[3]) if len(sys.argv) > 3 else 6),
     "check": lambda: check(space)}[cmd]()
