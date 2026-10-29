"""DMA timing models. complete(dma, inflight, t) returns the completion time in cycles."""

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
