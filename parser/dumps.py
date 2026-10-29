"""Locate artifacts in a record's dump directory and normalize bundle text."""

import glob
import os
import re

COMMENT = re.compile(r"/\*.*?\*/", re.S)


def final_bundles_path(dump_root, kernel_tag):
    paths = glob.glob(os.path.join(dump_root, "jf", f"*-{kernel_tag}.*-final_bundles.txt"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one final_bundles for {kernel_tag} in {dump_root}: {paths}")
    return paths[0]


def bundle_lines(text):
    """Bundle lines with comments removed, so annotation-only differences compare equal."""
    body = COMMENT.sub("", text)
    return [" ".join(line.split()) for line in body.splitlines() if re.match(r"\s*(0x)?[0-9a-f]+\s+:", line)]


def static_estimated_cycles(dump_root, kernel_tag):
    paths = glob.glob(os.path.join(dump_root, "static_profile", f"{kernel_tag}.*static_llo_profile.textproto"))
    if not paths:
        return None
    m = re.search(r"estimated_cycles:\s*(\d+)", open(paths[0]).read())
    return int(m.group(1)) if m else None
