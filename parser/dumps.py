"""Locate artifacts in a record's dump directory and normalize bundle text."""

import glob
import os
import re

COMMENT = re.compile(r"/\*.*?\*/", re.S)


def final_bundles_path(dump_root, kernel_tag, exact=False):
    """exact: kernel_tag is a full HLO name ('k_add.1'); else a stem matched with any '.N' suffix."""
    pat = f"*-{glob.escape(kernel_tag)}-[0-9]*-final_bundles.txt" if exact else f"*-{kernel_tag}.*-final_bundles.txt"
    paths = [p for p in glob.glob(os.path.join(dump_root, "jf", pat)) if "schedule-analysis" not in p and "static_gap" not in p]
    if exact and not paths:  # a one-kernel program's dump may drop the '.N' the profiler shows
        return final_bundles_path(dump_root, re.sub(r"\.\d+$", "", kernel_tag))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one final_bundles for {kernel_tag} in {dump_root}: {paths}")
    return paths[0]


def bundle_lines(text):
    """Bundle lines with comments removed, so annotation-only differences compare equal."""
    body = COMMENT.sub("", text)
    return [" ".join(line.split()) for line in body.splitlines() if re.match(r"\s*(0x)?[0-9a-f]+\s+:", line)]


def static_estimated_cycles(dump_root, kernel_tag, exact=False):
    pat = f"{glob.escape(kernel_tag)}_*static_llo_profile.textproto" if exact else f"{kernel_tag}.*static_llo_profile.textproto"
    paths = glob.glob(os.path.join(dump_root, "static_profile", pat))
    if exact and not paths:
        return static_estimated_cycles(dump_root, re.sub(r"\.\d+$", "", kernel_tag))
    if not paths:
        return None
    m = re.search(r"estimated_cycles:\s*(\d+)", open(paths[0]).read())
    return int(m.group(1)) if m else None
