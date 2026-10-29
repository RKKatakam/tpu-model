"""Noise report from device records: within-run, process-to-process, session-to-session.

usage: noise_report.py BATCH_DIR AOT_SESSION_DIR > report.md
BATCH_DIR holds <session>/<record_id>/record.json; AOT_SESSION_DIR holds <kernel>__aot__full records.
"""

import glob
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import dumps  # noqa: E402

LINE = "XLA Ops"


def load(batch_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(batch_dir, "*", "*", "record.json"))):
        r = json.load(open(path))
        tag = r["config"]["kernel_tag"]
        d = os.path.dirname(path)
        bundles = open(dumps.final_bundles_path(os.path.join(d, "dump"), tag)).read()
        samples = r["timing"]["by_line"][LINE]["samples_ns"]
        rows.append(
            {
                "dir": d,
                "session": r["machine"]["session"],
                "hostname": r["machine"]["hostname"],
                "kernel": r["kernel"],
                "tag": tag,
                "process": r["record_id"].split("__p")[1],
                "samples": samples,
                "median": statistics.median(samples),
                "bundles_sha": hashlib.sha256(bundles.encode()).hexdigest()[:12],
                "est_cycles": dumps.static_estimated_cycles(os.path.join(d, "dump"), tag),
                "versions": r["versions"],
            }
        )
    return rows


def pct(x):
    return f"{100 * x:.2f}%"


def spread(values):
    return (max(values) - min(values)) / statistics.median(values)


def main(batch_dir, aot_dir):
    rows = load(batch_dir)
    out = print
    sessions = sorted({(r["session"], r["hostname"]) for r in rows})
    out("## Sessions\n")
    out("| session | hostname | records |")
    out("|---|---|---|")
    for s, h in sessions:
        out(f"| {s} | {h} | {sum(r['session'] == s for r in rows)} |")

    out("\n## Within-run spread (200 reps per process)\n")
    out("| kernel | session | proc | median µs | CV | (p90-p10)/med | MAD/med | max/med | last50/first50 |")
    out("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["kernel"], r["session"], r["process"])):
        s = sorted(r["samples"])
        n, med = len(s), r["median"]
        p10, p90 = s[int(0.1 * (n - 1))], s[int(0.9 * (n - 1))]
        mad = statistics.median(abs(x - med) for x in s)
        cv = statistics.stdev(s) / statistics.fmean(s)
        drift = statistics.median(r["samples"][-50:]) / statistics.median(r["samples"][:50]) - 1
        out(
            f"| {r['kernel']} | {r['session']} | {r['process']} | {med / 1e3:.3f} | {pct(cv)} | "
            f"{pct((p90 - p10) / med)} | {pct(mad / med)} | {max(s) / med:.3f} | {drift:+.2%} |"
        )

    by_kernel = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel"]].append(r)
    out("\n## Process-to-process and session-to-session spread of the median\n")
    out("Spread is (max - min) / median of the medians.\n")
    out("| kernel | median µs | process spread (worst session) | session spread | session medians µs | XLA est. cycles |")
    out("|---|---|---|---|---|---|")
    for k, rs in sorted(by_kernel.items()):
        per_session = defaultdict(list)
        for r in rs:
            per_session[r["session"]].append(r["median"])
        proc = max(spread(v) for v in per_session.values())
        sess_medians = {s: statistics.median(v) for s, v in sorted(per_session.items())}
        grand = statistics.median(r["median"] for r in rs)
        est = {r["est_cycles"] for r in rs}
        out(
            f"| {k} | {grand / 1e3:.3f} | {pct(proc)} | {pct(spread(list(sess_medians.values())))} | "
            f"{', '.join(f'{s}={m / 1e3:.3f}' for s, m in sess_medians.items())} | {', '.join(map(str, est))} |"
        )

    out("\n## Same binary everywhere (final bundles)\n")
    out("| kernel | distinct device bundle sha256 | AOT (CPU host) sha256 | AOT == device (bytes) | AOT == device (comment-stripped) |")
    out("|---|---|---|---|---|")
    for k, rs in sorted(by_kernel.items()):
        tag = rs[0]["tag"]
        aot = open(dumps.final_bundles_path(os.path.join(aot_dir, f"{k}__aot__full", "dump"), tag)).read()
        dev = open(dumps.final_bundles_path(os.path.join(rs[0]["dir"], "dump"), tag)).read()
        shas = sorted({r["bundles_sha"] for r in rs})
        out(
            f"| {k} | {', '.join(shas)} | {hashlib.sha256(aot.encode()).hexdigest()[:12]} | {aot == dev} | "
            f"{dumps.bundle_lines(aot) == dumps.bundle_lines(dev)} |"
        )
    versions = {json.dumps(r["versions"], sort_keys=True) for r in rows}
    out(f"\nVersions across all records: {versions}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
