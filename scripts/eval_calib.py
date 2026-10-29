"""Score a candidate core/DMA parameter set on the calibration split only.

usage: eval_calib.py CORE_JSON DMA_JSON BATCH_DIR [BATCH_DIR ...]
Prints median |err|, counts within 2% and 5%, worst, and the median signed error per kernel. Never
reads validation-split records, so it is safe to use while choosing between model variants.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.split import split_of  # noqa: E402
from model.operands import integer_operands  # noqa: E402
from model.predict import Params, load_core, load_dma, load_records, predict_ns, simulate  # noqa: E402

CLASSES = ("compute", "streaming", "latency")


def err(args):
    r, core_json, dma_json = args
    P = Params.load("model/params/phase1.json")
    prog = r.program()
    tr = simulate(prog, P, load_core(core_json), load_dma(dma_json), integer_operands(r, prog))
    return r.raw["record_id"], r.raw["kernel"], predict_ns(tr, P) / r.measured_ns() - 1


def main(core_json, dma_json, dirs):
    recs = [r for d in dirs for r in load_records(d)
            if split_of(r.raw["record_id"]) == "calib" and r.raw["kernel_class"] in CLASSES and r.raw["kernel"] != "flash"]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        res = list(pool.map(err, [(r, core_json, dma_json) for r in recs]))
    a = np.abs([e for _, _, e in res])
    by = {}
    for _, k, e in res:
        by.setdefault(k, []).append(e)
    fam = "  ".join(f"{k}:{np.median(v):+.1%}" for k, v in sorted(by.items()))
    print(f"{core_json} {dma_json}: n={len(a)} median {np.median(a):.2%} within2 {np.sum(a <= .02)} "
          f"within5 {np.sum(a <= .05)} worst {a.max():.1%} | {fam}")
    out = os.environ.get("TPUM_EVAL_OUT")
    if out:
        with open(out, "w") as f:
            json.dump({rid: e for rid, _, e in res}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
