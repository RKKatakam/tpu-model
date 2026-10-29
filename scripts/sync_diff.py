"""Compare the replay's DMA waits with the hardware's, wait by wait (diagnostic only).

usage: sync_diff.py BATCH_DIR RECORD_ID [N_SHOWN]
Needs a record captured with profiler option tpu_trace_mode=TRACE_COMPUTE_AND_SYNC: the profile's
"Tensor Core Sync Flag" line then has one SyncWait (blocked, with duration) or SyncNoWait event per
dma.done.wait. The median-duration rep is aligned with the replay by wait order (same control flow).
"""

import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture import xplane  # noqa: E402
from model.predict import Params, load_core, load_dma, load_records, predict_record_ns  # noqa: E402

CLOCK = 1.5


def hardware_waits(record):
    prof = xplane.load(xplane.find_xplane(os.path.join(record.dir, "profile")))
    ev = list(xplane.device_events(prof))
    reps = sorted((s, d) for _, line, n, s, d in ev if line == "XLA Ops" and any(f"%{h} = " in n for h in record.hlo_names))
    med = statistics.median(d for _, d in reps)
    k0, kd = min(reps, key=lambda sd: abs(sd[1] - med))
    waits = []
    for _, line, n, s, d in sorted(ev, key=lambda e: e[3]):
        if line == "Tensor Core Sync Flag" and k0 <= s <= k0 + kd and n.startswith(("SyncWait", "SyncNoWait")):
            waits.append((s - k0, n.split(":")[1], d if n.startswith("SyncWait") else 0.0))
    return kd, waits


def main(batch_dir, rid, shown=24):
    r = next(r for r in load_records(batch_dir) if r.raw["record_id"] == rid)
    kd, hw = hardware_waits(r)
    P = Params.load("model/params/phase1.json")
    ns, traces = predict_record_ns(r, P, load_core(), load_dma())
    mw = [(t / CLOCK, f, stall / CLOCK, pc) for tr in traces for pc, t, f, stall in tr.waits]
    print(f"{rid}: measured {kd:.0f} ns, model {ns:.0f} ns ({ns / kd - 1:+.1%}); waits hw {len(hw)} model {len(mw)}")
    n = min(len(hw), len(mw))
    hs, ms = sum(w[2] for w in hw[:n]), sum(w[2] for w in mw[:n])
    print(f"total wait stall: hw {hs:.0f} ns, model {ms:.0f} ns; non-wait time: hw {kd - hs:.0f} ns, model {ns - ms:.0f} ns")
    # per wait site (bundle index in the replay = position in the program)
    by_site = defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
    for (ht, hf, hstall), (mt, mf, mstall, pc) in zip(hw[:n], mw[:n]):
        s = by_site[pc]
        s[0] += 1; s[1] += hstall; s[2] += mstall; s[3] += hstall > 0; s[4] += mstall > 0
    print("per wait site: bundle  count  hw stall/wait  model stall/wait  hw blocked  model blocked")
    for pc, (c, h, m, hb, mb) in sorted(by_site.items(), key=lambda kv: -abs(kv[1][1] - kv[1][2])):
        print(f"   {pc:6d} {c:6d} {h / c:12.1f} ns {m / c:14.1f} ns {hb:9d} {mb:12d}")
    print("first waits (hw time, flag, stall | model time, stall):")
    for i in range(min(shown, n)):
        (ht, hf, hstall), (mt, mf, mstall, pc) = hw[i], mw[i]
        print(f"   {i:4d} hw t={ht:9.0f} flag {hf:>4s} stall {hstall:7.0f} | model t={mt:9.0f} stall {mstall:7.0f} (bundle {pc})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 24)
