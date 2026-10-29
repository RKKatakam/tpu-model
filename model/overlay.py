"""Instruction overlays (profiler sync flag 511 "sequencer_overlay"; bench phase3s/phase3t).

A program of at most `capacity_bundles` bundles is loaded whole before the kernel starts and costs
nothing here. A larger one runs out of `slots` overlay buffers. Its chunks are `chunk_bundles`
bundles on a grid from bundle 0, also split at the targets of forward branches into another chunk.
Entering a chunk costs `switch_cycles`. If the chunk is not resident, the core first waits for its
load, which takes `load_cycles` whatever the chunk's length (a 17-bundle chunk loads as slowly as a
full one) and barely slows under the kernel's own HBM traffic (philox: 8.7 vs 7.8 us), so it does not
go through the DMA model. After entering chunk j, the sequencer prefetches
the next chunk: j + 1, or after the last chunk the one holding the target of the program's last
backward branch (the outer loop head). A load evicts the least recently entered resident chunk.
The sequencer predicts branches statically, backward taken and forward not taken: a forward branch
into another chunk waits only when taken, while a backward one (a loop exit) waits for its target
chunk even when not taken.
"""

import re


class Overlays:
    def __init__(self, program, p):
        n = len(program.bundles)
        self.on = bool(p) and n > p["capacity_bundles"]
        if not self.on:
            return
        size = int(p["chunk_bundles"])
        self.switch = p["switch_cycles"]
        self.load_cycles = p["load_cycles"]
        self.slots = p["slots"]
        branches = []  # (source bundle, target bundle)
        for b in program.bundles:
            for op in b.ops:
                if op.mnemonic == "sbr.rel":
                    region = int(re.search(r"\$region(\d+)", op.args[0]).group(1))
                    branches.append((b.index, program.target(region)))
        starts = set(range(0, n, size))
        starts |= {t for s, t in branches if s // size != t // size and s < t < n}
        self.start = sorted(starts)
        self.chunk_of = [0] * n
        for c, (a, e) in enumerate(zip(self.start, self.start[1:] + [n])):
            self.chunk_of[a:e] = [c] * (e - a)
        back = [t for s, t in branches if t <= s]
        self.wrap = self.chunk_of[back[-1]] if back else 0
        self.cur = 0
        # resident chunk -> cycle its load completes (loaded before the kernel: 0), in entry order
        self.resident = {c: 0.0 for c in range(min(self.slots, len(self.start)))}

    def _load(self, chunk, t):
        while len(self.resident) >= self.slots:
            del self.resident[next(c for c in self.resident if c != self.cur)]
        self.resident[chunk] = t + self.load_cycles

    def not_taken(self, pc, target, t):
        """Cycle after a not-taken branch from `pc` to `target`: a backward one waits for its chunk."""
        chunk = self.chunk_of[target] if target < len(self.chunk_of) else self.cur
        if target > pc or chunk == self.cur:
            return t
        if chunk not in self.resident:
            self._load(chunk, t)
        return max(t, self.resident[chunk])

    def enter(self, pc, t):
        """Cycle at which bundle `pc` can issue when the front end reaches it at `t`."""
        chunk = self.chunk_of[pc]
        if chunk == self.cur:
            return t
        if chunk not in self.resident:
            self._load(chunk, t)
        t = max(t, self.resident.pop(chunk))
        self.resident[chunk] = t  # most recently entered
        self.cur = chunk
        nxt = chunk + 1 if chunk + 1 < len(self.start) else self.wrap
        if nxt not in self.resident:
            self._load(nxt, t)
        return t + self.switch
