"""DMA timing models. complete(dma, inflight, t) returns the completion time in cycles."""

import bisect
import math

GRANULE_BYTES = 32


class SerialHbm:
    """Fixed issue latency, then transfers share one HBM pipe FIFO at a fixed bandwidth."""

    def __init__(self, latency_cycles, bytes_per_cycle):
        self.latency = latency_cycles
        self.bw = bytes_per_cycle
        self.free = 0.0

    def complete(self, d, inflight, t):
        start = max(t + self.latency, self.free)
        self.free = start + d.granules * GRANULE_BYTES / self.bw
        return self.free


class TableHbm:
    """Single-stream copy time T(bytes) measured by bench dma_seq, split into a latency part
    L = T(smallest) that overlaps across transfers and a transfer part T - L that occupies
    one shared HBM pipe in FIFO order. A lone transfer reproduces T(bytes) exactly."""

    def __init__(self, table_cycles):
        pts = sorted(table_cycles)
        self.logb = [math.log(b) for b, _ in pts]
        self.t = [c for _, c in pts]
        self.latency = self.t[0]
        self.free = 0.0

    def copy_cycles(self, nbytes):
        x = math.log(max(nbytes, 1))
        i = bisect.bisect_left(self.logb, x)
        if i <= 0:
            return self.t[0]
        if i >= len(self.t):
            (x0, x1), (y0, y1) = self.logb[-2:], self.t[-2:]
            slope = (y1 - y0) / (math.exp(x1) - math.exp(x0))
            return y1 + slope * (nbytes - math.exp(x1))
        x0, x1, y0, y1 = self.logb[i - 1], self.logb[i], self.t[i - 1], self.t[i]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def complete(self, d, inflight, t):
        transfer = self.copy_cycles(d.granules * GRANULE_BYTES) - self.latency
        start = max(t + self.latency, self.free)
        self.free = start + transfer
        return self.free
