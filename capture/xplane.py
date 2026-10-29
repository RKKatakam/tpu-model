"""Extract per-rep device time of a named kernel from a .xplane.pb.

Event field names (plane.lines, line.events, ev.duration_ns) are unverified
for this JAX version, so summarize() records every plane/line/event name seen
and the selection rule can be audited against the raw profile.
"""

import glob
import os
import statistics
from collections import defaultdict


def find_xplane(logdir):
    paths = glob.glob(os.path.join(logdir, "plugins", "profile", "*", "*.xplane.pb"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one .xplane.pb under {logdir}, found {paths}")
    return paths[0]


def load(path):
    from jax.profiler import ProfileData

    return ProfileData.from_file(path)


def device_events(profile):
    """Yield (plane, line, name, start_ns, duration_ns) for TPU device planes."""
    for plane in profile.planes:
        if not plane.name.startswith("/device:TPU"):
            continue
        for line in plane.lines:
            for ev in line.events:
                yield plane.name, line.name, ev.name, ev.start_ns, ev.duration_ns


def summarize(profile):
    groups = defaultdict(list)
    planes = {}
    for plane in profile.planes:
        planes[plane.name] = sorted({line.name for line in plane.lines})
    for plane, line, name, _, dur in device_events(profile):
        groups[(plane, line, name)].append(dur)
    events = [
        {
            "plane": p,
            "line": l,
            "name": n,
            "count": len(d),
            "median_ns": statistics.median(d),
        }
        for (p, l, n), d in sorted(groups.items())
    ]
    return {"planes": planes, "device_events": events}


def kernel_durations(profile, kernel_tag):
    """Per line, durations in start order of events whose name contains kernel_tag."""
    hits = defaultdict(list)
    for _, line, name, start, dur in device_events(profile):
        if kernel_tag in name:
            hits[line].append((start, dur))
    return {line: [dur for _, dur in sorted(h)] for line, h in hits.items()}


def custom_call_name(summary):
    """HLO name (e.g. 'flash_attention.1') of the one tpu_custom_call op in a summarize() result.
    Dump files and profiler events are named by it, not by the Pallas kernel function."""
    names = {e["name"].split(" = ")[0].lstrip("%") for e in summary["device_events"]
             if e["line"] == "XLA Ops" and 'custom_call_target="tpu_custom_call"' in e["name"]}
    if len(names) != 1:
        raise ValueError(f"expected one tpu_custom_call op, found {sorted(names)}")
    return names.pop()


def stats(samples):
    s = sorted(samples)
    n = len(s)
    med = statistics.median(s)

    def q(p):
        return s[min(n - 1, int(round(p * (n - 1))))]

    return {
        "n": n,
        "median_ns": med,
        "mean_ns": statistics.fmean(s),
        "stdev_ns": statistics.stdev(s) if n > 1 else 0.0,
        "min_ns": s[0],
        "max_ns": s[-1],
        "p10_ns": q(0.10),
        "p90_ns": q(0.90),
        "mad_ns": statistics.median(abs(x - med) for x in s),
    }
