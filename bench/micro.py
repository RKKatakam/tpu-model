"""Microbenchmark kernels. Each isolates one mechanism; the bundle check that
confirms it lives in bench/checks.py.

Inputs and outputs stay in HBM (memory_space=HBM) so the pipeline emits no
DMAs; work happens on VMEM scratch whose contents are irrelevant to timing.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HBM = pl.BlockSpec(memory_space=pltpu.HBM)


def _no_dma_call(kernel, scratch, name):
    """pallas_call with one untouched HBM input/output and the given scratch."""

    def fn(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[HBM],
            out_specs=HBM,
            scratch_shapes=scratch,
            name=name,
        )(x)

    return fn, [jax.ShapeDtypeStruct((8, 128), jnp.float32)]


def valu_straight(params):
    """k rounds of vadd.f32 on nv independent vregs; straight-line, no loop, no DMA.

    isolates: bundle issue rate (clock) on vector-ALU-only code.
    """
    k, nv = params["k"], params["nv"]

    def kernel(x_hbm, o_hbm, buf):
        vs = [buf[pl.ds(8 * i, 8), :] for i in range(nv)]
        c = buf[pl.ds(8 * nv, 8), :]
        for _ in range(k):
            vs = [v + c for v in vs]
        for i, v in enumerate(vs):
            buf[pl.ds(8 * i, 8), :] = v

    return _no_dma_call(kernel, [pltpu.VMEM((8 * (nv + 1), 128), jnp.float32)], "k_valu_straight")


def valu_loop(params):
    """fori_loop of `trip` iterations, each k rounds of vadd on nv vregs carried in registers.

    isolates: loop overhead per iteration on top of the straight-line body.
    """
    trip, k, nv = params["trip"], params["k"], params["nv"]

    def kernel(x_hbm, o_hbm, buf):
        c = buf[pl.ds(8 * nv, 8), :]

        def body(_, vs):
            for _ in range(k):
                vs = tuple(v + c for v in vs)
            return vs

        vs = jax.lax.fori_loop(0, trip, body, tuple(buf[pl.ds(8 * i, 8), :] for i in range(nv)))
        for i, v in enumerate(vs):
            buf[pl.ds(8 * i, 8), :] = v

    return _no_dma_call(kernel, [pltpu.VMEM((8 * (nv + 1), 128), jnp.float32)], "k_valu_loop")


def dma_seq(params):
    """`reps` sequential HBM->VMEM copies of x[0:rows] (rows x 128 f32), each started then waited.

    isolates: single-stream DMA latency + transfer time vs size.
    """
    rows, reps, total_rows = params["rows"], params["reps"], params["total_rows"]

    def kernel(x_hbm, o_hbm, buf, sem):
        for _ in range(reps):
            cp = pltpu.make_async_copy(x_hbm.at[pl.ds(0, rows), :], buf, sem)
            cp.start()
            cp.wait()

    def fn(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            in_specs=[HBM],
            out_specs=HBM,
            scratch_shapes=[pltpu.VMEM((rows, 128), jnp.float32), pltpu.SemaphoreType.DMA],
            name="k_dma_seq",
        )(x)

    return fn, [jax.ShapeDtypeStruct((total_rows, 128), jnp.float32)]


VOPS = {
    "add": lambda v, c: v + c,
    "mul": lambda v, c: v * c,
    "max": lambda v, c: jnp.maximum(v, c),
    "exp": lambda v, c: jnp.exp(v),
    "roll_lane": lambda v, c: pltpu.roll(v, 1, 1),
    "roll_sublane": lambda v, c: pltpu.roll(v, 1, 0),
    "sum_lane": lambda v, c: v + jnp.sum(v, axis=1, keepdims=True),
    "cast": lambda v, c: v.astype(jnp.bfloat16).astype(jnp.float32),
}


def vop_chain(params):
    """k rounds of one vector op on nv vregs, straight-line, no DMA.

    nv=1 is a dependent chain (exposes result latency); nv=8 exposes issue throughput.
    isolates: latency/throughput of one VPU/EUP/XLU op class.
    """
    k, nv, opname = params["k"], params["nv"], params["op"]
    op = VOPS[opname]

    def kernel(x_hbm, o_hbm, buf):
        vs = [buf[pl.ds(8 * i, 8), :] for i in range(nv)]
        c = buf[pl.ds(8 * nv, 8), :]
        for _ in range(k):
            vs = [op(v, c) for v in vs]
        for i, v in enumerate(vs):
            buf[pl.ds(8 * i, 8), :] = v

    return _no_dma_call(kernel, [pltpu.VMEM((8 * (nv + 1), 128), jnp.float32)], "k_vop_chain")


def mem_chain(params):
    """k rounds of buf = buf + c through VMEM (vld -> vadd -> vst -> next vld of the same address).

    isolates: store-to-load round trip through VMEM.
    """
    k = params["k"]

    def kernel(x_hbm, o_hbm, buf):
        c = buf[pl.ds(8, 8), :]
        for _ in range(k):
            buf[pl.ds(0, 8), :] = buf[pl.ds(0, 8), :] + c

    return _no_dma_call(kernel, [pltpu.VMEM((16, 128), jnp.float32)], "k_mem_chain")


def mxu_dot(params):
    """reps x (acc += x @ w) with x (m,k), w (k,n) in VMEM scratch (dtypes `lhs`, `rhs`, default
    bf16), acc f32.

    isolates: MXU weight push, LHS streaming, result pop; no DMA.
    """
    m, kk, n, reps = params["m"], params["k"], params["n"], params["reps"]
    dt = {"bf16": jnp.bfloat16, "f32": jnp.float32}
    lhs, rhs = dt[params.get("lhs", "bf16")], dt[params.get("rhs", "bf16")]

    def kernel(x_hbm, o_hbm, xs, ws, acc):
        def body(_, carry):
            acc[...] += jnp.dot(xs[...], ws[...], preferred_element_type=jnp.float32)
            return carry

        jax.lax.fori_loop(0, reps, body, 0)

    scratch = [
        pltpu.VMEM((m, kk), lhs),
        pltpu.VMEM((kk, n), rhs),
        pltpu.VMEM((m, n), jnp.float32),
    ]
    return _no_dma_call(kernel, scratch, "k_mxu_dot")


def _dma_call(kernel, src_shape, scratch, name):
    def fn(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(src_shape, jnp.float32),
            in_specs=[HBM],
            out_specs=HBM,
            scratch_shapes=scratch,
            name=name,
        )(x)

    return fn, [jax.ShapeDtypeStruct(src_shape, jnp.float32)]


def dma_out(params):
    """`reps` sequential VMEM->HBM copies of (rows,128) f32, each started then waited.

    isolates: single-stream write DMA vs size.
    """
    rows, reps = params["rows"], params["reps"]

    def kernel(x_hbm, o_hbm, buf, sem):
        for _ in range(reps):
            cp = pltpu.make_async_copy(buf, o_hbm.at[pl.ds(0, rows), :], sem)
            cp.start()
            cp.wait()

    return _dma_call(kernel, (8192, 128), [pltpu.VMEM((rows, 128), jnp.float32), pltpu.SemaphoreType.DMA], "k_dma_out")


def dma_conc(params):
    """`reps` rounds of: start `streams` HBM->VMEM copies of (rows,128) f32 from disjoint HBM
    regions into disjoint buffers with their own semaphores, then wait for all.

    isolates: concurrent read DMAs (latency overlap, bandwidth sharing).
    """
    rows, streams, reps = params["rows"], params["streams"], params["reps"]

    def kernel(x_hbm, o_hbm, buf, sems):
        for _ in range(reps):
            cps = [
                pltpu.make_async_copy(x_hbm.at[pl.ds(i * rows, rows), :], buf.at[i], sems.at[i])
                for i in range(streams)
            ]
            for cp in cps:
                cp.start()
            for cp in cps:
                cp.wait()

    scratch = [pltpu.VMEM((streams, rows, 128), jnp.float32), pltpu.SemaphoreType.DMA((streams,))]
    return _dma_call(kernel, (max(8192, streams * rows), 128), scratch, "k_dma_conc")


def dma_inout(params):
    """`reps` rounds of one HBM->VMEM and one VMEM->HBM copy in flight together.

    isolates: read/write DMA interaction.
    """
    rows, reps = params["rows"], params["reps"]

    def kernel(x_hbm, o_hbm, buf, sems):
        for _ in range(reps):
            rd = pltpu.make_async_copy(x_hbm.at[pl.ds(0, rows), :], buf.at[0], sems.at[0])
            wr = pltpu.make_async_copy(buf.at[1], o_hbm.at[pl.ds(rows, rows), :], sems.at[1])
            rd.start()
            wr.start()
            rd.wait()
            wr.wait()

    scratch = [pltpu.VMEM((2, rows, 128), jnp.float32), pltpu.SemaphoreType.DMA((2,))]
    return _dma_call(kernel, (max(8192, 2 * rows), 128), scratch, "k_dma_inout")


def dma_strided(params):
    """`reps` sequential copies of x[0:rows, 0:128] from a (rows, width) f32 array: each row is a
    512 B contiguous piece at a stride of width*4 bytes.

    isolates: strided read DMA cost vs row count and stride.
    """
    rows, width, reps = params["rows"], params["width"], params["reps"]

    def kernel(x_hbm, o_hbm, buf, sem):
        for _ in range(reps):
            cp = pltpu.make_async_copy(x_hbm.at[:, pl.ds(0, 128)], buf, sem)
            cp.start()
            cp.wait()

    def fn(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            in_specs=[HBM],
            out_specs=HBM,
            scratch_shapes=[pltpu.VMEM((rows, 128), jnp.float32), pltpu.SemaphoreType.DMA],
            name="k_dma_strided",
        )(x)

    return fn, [jax.ShapeDtypeStruct((rows, width), jnp.float32)]


def dma_stream(params):
    """Pallas-pipelined stream over a (rows, cols) f32 array with `reads` inputs and `writes`
    outputs per grid step and only vld/vst/vadd as compute.

    reads=1,writes=0 keeps the output block resident (written once at the end);
    reads=0 writes a constant. isolates: sustained read/write mix under the pipeline emitter.
    """
    rows, cols, br = params["rows"], params["cols"], params["block_rows"]
    nr, nw = params["reads"], params["writes"]
    blk = pl.BlockSpec((br, cols), lambda i: (i, 0))
    resident = pl.BlockSpec((8, cols), lambda i: (0, 0))

    def kernel(*refs):
        ins, outs = refs[:nr], refs[nr:]
        if nr == 0:
            val = jnp.full((br, cols), 1.0, jnp.float32)
        else:
            val = ins[0][...]
            for r in ins[1:]:
                val = val + r[...]
        if nw == 0:
            outs[0][...] = val[:8]
        else:
            for o in outs:
                o[...] = val

    out_shape = [jax.ShapeDtypeStruct((rows, cols) if nw else (8, cols), jnp.float32) for _ in range(max(nw, 1))]
    out_specs = [blk if nw else resident for _ in range(max(nw, 1))]

    def fn(*xs):
        res = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid=(rows // br,),
            in_specs=[blk] * nr,
            out_specs=out_specs,
            name="k_dma_stream",
        )(*xs)
        return res

    specs = [jax.ShapeDtypeStruct((rows, cols), jnp.float32) for _ in range(nr)]
    if not specs:
        # pallas_call needs no inputs here, but jit needs an argument to key the call on.
        def fn0(dummy):
            return fn()
        return fn0, [jax.ShapeDtypeStruct((8, 128), jnp.float32)]
    return fn, specs


def dma_gather(params):
    """`n` copies of one (rows, 128) f32 page of a (pages, rows, 128) HBM array into VMEM, at page
    indices read from SMEM (scalar prefetch), keeping `inflight` copies outstanding. order 'seq'
    reads pages 0..n-1, 'random' a fixed random permutation.

    isolates: small-DMA throughput vs address pattern and concurrency (paged attention's page loads).
    """
    rows, n, c, pages = params["rows"], params["n"], params["inflight"], params["pages"]

    def kernel(idx_ref, x_hbm, o_ref, buf, sems):
        def copy(i, slot):
            return pltpu.make_async_copy(x_hbm.at[idx_ref[i]], buf.at[slot], sems.at[slot])

        for s in range(c):
            copy(s, s).start()

        def body(i, carry):
            slot = i % c
            copy(i, slot).wait()

            @pl.when(i + c < n)
            def _():
                copy(i + c, slot).start()

            return carry

        jax.lax.fori_loop(0, n, body, 0)
        o_ref[...] = buf[0]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(1,), in_specs=[HBM],
        out_specs=pl.BlockSpec((rows, 128), lambda i, idx: (0, 0)),
        scratch_shapes=[pltpu.VMEM((c, rows, 128), jnp.float32), pltpu.SemaphoreType.DMA((c,))],
    )

    def fn(idx, x):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((rows, 128), jnp.float32),
                              grid_spec=grid_spec, name="k_dma_gather")(idx, x)

    specs = [jax.ShapeDtypeStruct((n,), jnp.int32), jax.ShapeDtypeStruct((pages, rows, 128), jnp.float32)]

    def make_inputs(seed):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(pages)[:n] if params["order"] == "random" else np.arange(n)
        return [idx.astype(np.int32), np.zeros((pages, rows, 128), np.float32)]

    return fn, specs, make_inputs


MICRO = {
    "dma_gather": (dma_gather, "micro"),
    "dma_stream": (dma_stream, "micro"),
    "vop_chain": (vop_chain, "micro"),
    "mem_chain": (mem_chain, "micro"),
    "mxu_dot": (mxu_dot, "micro"),
    "dma_out": (dma_out, "micro"),
    "dma_conc": (dma_conc, "micro"),
    "dma_inout": (dma_inout, "micro"),
    "dma_strided": (dma_strided, "micro"),
    "valu_straight": (valu_straight, "micro"),
    "valu_loop": (valu_loop, "micro"),
    "dma_seq": (dma_seq, "micro"),
}
