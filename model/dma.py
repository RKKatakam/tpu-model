"""DMA timing models. The machine calls issue(d, t) when a DMA starts and project(t) before
it reads completion times at a wait; project assumes no further arrivals, which holds
because the core is stalled at the wait."""

import bisect
import math

GRANULE_BYTES = 32


class Curve:
    """Single-stream copy time vs bytes, log-size interpolation, linear extrapolation above."""

    def __init__(self, points):
        pts = sorted(points)
        self.logb = [math.log(b) for b, _ in pts]
        self.t = [c for _, c in pts]

    def __call__(self, nbytes):
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


class TableHbm:
    """Single-stream table; latency overlaps, transfer parts serialize on one FIFO pipe."""

    def __init__(self, table_cycles):
        self.curve = Curve(table_cycles)
        self.latency = self.curve.t[0]
        self.free = 0.0

    def copy_cycles(self, nbytes):
        return self.curve(nbytes)

    def issue(self, d, t):
        transfer = self.curve(d.granules * GRANULE_BYTES) - self.latency
        start = max(t + self.latency, self.free)
        self.free = start + transfer
        d.done = self.free

    def project(self, t):
        pass


class _Xfer:
    __slots__ = ("d", "ready", "left", "cap", "write", "post", "paused")

    def __init__(self, d, ready, nbytes, cap, write):
        self.d, self.ready, self.left, self.cap, self.write = d, ready, float(nbytes), cap, write
        self.paused = 0.0  # cycles of start-up already paused (pause_latency)
        self.post = math.inf  # posted completion time (writes), if earlier than the data finishing


class SharedHbm:
    """Processor sharing over one aggregate HBM bandwidth.

    A transfer of s bytes waits `issue_cycles` on a serial descriptor engine, then a start-up
    latency L(s) = T_single(s) - s / bw (direction-specific curve and bandwidth), which overlaps
    with other transfers, then moves its bytes at a fair share of the aggregate: `bw_read` when
    only reads move, `bw_write` when only writes move, `bw_mixed` otherwise. A transfer that
    starts while the other direction is in flight pays `mix_cycles` more start-up. A lone transfer
    reproduces its measured single-stream time exactly."""

    def __init__(self, read_cycles, write_cycles, bw_read, bw_write, bw_mixed, issue_cycles, mix_cycles=0.0):
        self.curves = {False: Curve(read_cycles), True: Curve(write_cycles)}
        self.bw = {False: bw_read, True: bw_write}
        self.bw_mixed, self.issue_cycles, self.mix_cycles = bw_mixed, issue_cycles, mix_cycles
        self.engine_free = 0.0
        self.active = []
        self.now = 0.0
        self.lat_adj = {False: 0.0, True: 0.0}
        self.tail = {False: 0.0, True: 0.0}  # completion delay after the last byte (not occupying the pipe)

    def issue(self, d, t):
        self._advance(t)
        write = d.kind.endswith("to_hbm")
        nbytes = d.granules * GRANULE_BYTES
        start = max(t, self.engine_free)
        self.engine_free = start + self.issue_cycles
        # start-up = single-stream time - transfer - adjustment - tail (the tail comes after the data)
        latency = max(0.0, self.curves[write](nbytes) - nbytes / self.bw[write] - self.lat_adj[write]
                      - self.tail[write])
        if any(x.write != write and x.left > 0 for x in self.active):
            latency += self.mix_cycles
        # each contiguous piece of a strided copy whose HBM pieces are 8+ pieces apart costs
        # `stride_piece_bytes` of extra transfer (bench dma_strided, dma_mxu: ~0.65 ns per piece at
        # stride ratios 8-32; none at 2; ratio 4 unmeasured, calibration add says none)
        moved = nbytes
        if getattr(d, "stride_ratio", 1.0) >= 8 and getattr(d, "piece_granules", 0) > 0:
            # pieces cost more while another strided copy of the same direction is moving (interleaved)
            busy = any(x.write == write and x.left > 0 and getattr(x.d, "stride_ratio", 1.0) >= 8 for x in self.active)
            per = getattr(self, "stride_piece_bytes_conc", None) if busy else None
            moved += d.granules / d.piece_granules * (per if per is not None else getattr(self, "stride_piece_bytes", 0.0))
        x = _Xfer(d, start + latency, moved, math.inf, write)
        if write and getattr(self, "bw_post", None):
            x.post = x.ready + nbytes / self.bw_post
        self.active.append(x)
        self.project(t)

    def project(self, t):
        self._advance(t)
        state = [(x, x.ready, x.left, x.paused) for x in self.active]
        done = self._run(list(self.active), self.now, until=math.inf)
        for x, ready, left, paused in state:
            x.ready, x.left, x.paused = ready, left, paused
        for x in self.active:
            x.d.done = min(done[id(x)] + self.tail[x.write], x.post)

    def _rates(self, xs, now):
        moving = [x for x in xs if x.ready <= now and x.left > 0]
        if not moving:
            return moving, {}
        writes = sum(x.write for x in moving)
        if writes == 0:
            cap_total = self.bw[False]
        elif writes == len(moving):
            cap_total = self.bw[True]
        else:
            cap_total = self.bw_mixed
        # water-filling: fair share of the aggregate, capped per stream
        rates, pool, left = {}, list(moving), cap_total
        while pool:
            share = left / len(pool)
            capped = [x for x in pool if x.cap <= share]
            if not capped:
                for x in pool:
                    rates[id(x)] = share
                break
            for x in capped:
                rates[id(x)] = x.cap
                left -= x.cap
                pool.remove(x)
        return moving, rates

    def _run(self, xs, now, until):
        """Serve transfers from `now` until `until` (or all done); mutates xs; returns done times."""
        done = {}
        pause = getattr(self, "pause_latency", False)
        cap = getattr(self, "pause_cap", math.inf)
        while xs and now < until:
            moving, rates = self._rates(xs, now)
            dirs = {x.write for x in moving}
            # with pause_latency, a transfer's start-up does not progress while the other direction moves data
            paused = [x for x in xs if pause and x.ready > now and (not x.write) in dirs
                      and (pause != "reads" or not x.write) and x.paused < cap - 1e-9]
            nxt_ready = min((x.ready for x in xs if x.ready > now and x not in paused), default=math.inf)
            nxt_ready = min([nxt_ready] + [now + cap - x.paused for x in paused])
            nxt_done = min((now + x.left / rates[id(x)] for x in moving), default=math.inf)
            step_to = min(nxt_ready, nxt_done, until)
            if step_to == math.inf:
                break
            for x in moving:
                x.left -= rates[id(x)] * (step_to - now)
            for x in paused:
                x.ready += step_to - now
                x.paused = min(cap, x.paused + step_to - now)
                if cap - x.paused < 1e-9:  # float rounding would otherwise leave zero-length steps
                    x.paused = cap
            now = step_to
            for x in [x for x in xs if x.ready <= now and x.left <= 1e-6]:
                done[id(x)] = now
                xs.remove(x)
        return done

    def _advance(self, t):
        if t <= self.now:
            return
        finished = self._run(self.active, self.now, until=t)
        for x in list(self.active):
            if id(x) in finished:
                x.d.done = min(finished[id(x)] + self.tail[x.write], x.post)
        self.now = t


class DirFifoHbm(SharedHbm):
    """One FIFO queue per direction. Start-up latency L(s) = T_single(s) - s / bw_dir overlaps
    freely; bytes move in issue order within a direction. The two queue heads interleave: alone, a
    head moves at its direction's bandwidth; together each moves at 1 / (1/bw_read + 1/bw_write +
    turnaround) bytes per cycle (bench dma_stream: reads cost 1/bw_read, writes 1/bw_write, and
    overlapped bytes pay `turnaround` cycles per byte pair). A transfer issued while the other
    direction is in flight pays `mix_cycles` more start-up."""

    def __init__(self, read_cycles, write_cycles, bw_read, bw_write, turnaround, issue_cycles,
                 mix_cycles=0.0, read_lat_adj=0.0, write_lat_adj=0.0, single_queue=False, bw_post=None,
                 wake_cycles=0.0, pause_latency=False, read_tail=0.0, write_tail=0.0, pause_cap=math.inf):
        super().__init__(read_cycles, write_cycles, bw_read, bw_write, None, issue_cycles, mix_cycles)
        self.pause_latency = pause_latency
        self.pause_cap = pause_cap  # most cycles a start-up pauses (pause_latency)
        # part of each direction's single-stream latency comes after the data moves (completion
        # signalling): sequential copies see all of it, pipelined copies overlap the tail
        self.tail = {False: read_tail, True: write_tail}
        self.single_queue = single_queue
        # Semaphore wake-up: paid by the core only when a dma.done.wait actually blocks (Machine).
        # The single-stream curves come from start-then-wait loops, so they include it once per copy.
        self.wake_cycles = wake_cycles
        # Posted writes: the semaphore fires when the data has left VMEM (at bw_post) while HBM keeps
        # absorbing it at bw_write; None = completion when the last byte is written.
        self.bw_post = bw_post
        self.pair_rate = 1.0 / (1.0 / bw_read + 1.0 / bw_write + turnaround)
        # start-up latency = single-stream time - bytes / bw - adj (adj fitted: back-to-back
        # per-copy times include more than a pipelined transfer waits for)
        self.lat_adj = {False: read_lat_adj + wake_cycles, True: write_lat_adj + wake_cycles}

    single_queue = False  # one FIFO for both directions (set by DirFifoHbm(..., single_queue=True))

    def _rates(self, xs, now):
        heads = {}
        for x in xs:  # xs is in issue order
            if x.left > 0 and x.write not in heads:
                heads[x.write] = x
        if self.single_queue and heads:
            first = min(heads.values(), key=xs.index)
            heads = {first.write: first}
        moving = [x for x in heads.values() if x.ready <= now]
        if len(moving) == 2:
            return moving, {id(x): self.pair_rate for x in moving}
        return moving, {id(x): self.bw[x.write] for x in moving}


class NoDma:
    """For microbenchmarks that must not DMA."""

    def issue(self, d, t):
        raise AssertionError("unexpected DMA")

    def project(self, t):
        pass


class FifoHbm:
    """Same start-up latency and bandwidths as SharedHbm, but bytes are served first-in
    first-out: a transfer gets the whole aggregate bandwidth once those issued before it are
    done. Writes and reads share the pipe; a transfer whose service overlaps the other
    direction's in-flight start-up uses `bw_mixed` and pays `mix_cycles`."""

    def __init__(self, read_cycles, write_cycles, bw_read, bw_write, bw_mixed, issue_cycles, mix_cycles=0.0):
        self.curves = {False: Curve(read_cycles), True: Curve(write_cycles)}
        self.bw = {False: bw_read, True: bw_write}
        self.bw_mixed, self.issue_cycles, self.mix_cycles = bw_mixed, issue_cycles, mix_cycles
        self.engine_free = 0.0
        self.pipe_free = 0.0
        self.recent = []  # (direction, busy_until) of queued transfers

    def issue(self, d, t):
        write = d.kind.endswith("to_hbm")
        nbytes = d.granules * GRANULE_BYTES
        start = max(t, self.engine_free)
        self.engine_free = start + self.issue_cycles
        latency = max(0.0, self.curves[write](nbytes) - nbytes / self.bw[write])
        self.recent = [(w, until) for w, until in self.recent if until > start]
        mixed = any(w != write for w, _ in self.recent)
        bw = self.bw_mixed if mixed else self.bw[write]
        if mixed:
            latency += self.mix_cycles
        begin = max(start + latency, self.pipe_free)
        self.pipe_free = begin + nbytes / bw
        self.recent.append((write, self.pipe_free))
        d.done = self.pipe_free

    def project(self, t):
        pass
