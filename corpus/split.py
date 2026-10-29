"""Pre-registered corpus split, fixed before any corpus timing was seen.

A record is 'valid' (report only, never used to fit or diagnose-then-tune) when the first byte
of sha256(record_id) is odd; otherwise 'calib'.
"""

import hashlib


def split_of(record_id):
    return "valid" if hashlib.sha256(record_id.encode()).digest()[0] % 2 else "calib"
