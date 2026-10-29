"""Compare a kernel's final bundles between two record dirs.

usage: compare_bundles.py KERNEL_TAG RECORD_DIR_A RECORD_DIR_B
Prints byte equality, comment-stripped equality, bundle counts, and the first differing bundle.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import dumps  # noqa: E402


def main(tag, a, b):
    pa, pb = (dumps.final_bundles_path(os.path.join(d, "dump"), tag) for d in (a, b))
    ta, tb = open(pa).read(), open(pb).read()
    la, lb = dumps.bundle_lines(ta), dumps.bundle_lines(tb)
    print(f"{tag}: bytes_equal={ta == tb} stripped_equal={la == lb} bundles={len(la)} vs {len(lb)}")
    print(f"  a: {os.path.basename(pa)}\n  b: {os.path.basename(pb)}")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            print(f"  first diff at bundle {i}:\n    a: {x[:200]}\n    b: {y[:200]}")
            break
    return la == lb


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:4]) else 1)
