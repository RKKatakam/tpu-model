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
    bf16; `rhs_t` stores w as (n,k), as splash does for K), acc f32.

    isolates: MXU weight push, LHS streaming, result pop; no DMA.
    """
    m, kk, n, reps = params["m"], params["k"], params["n"], params["reps"]
    dt = {"bf16": jnp.bfloat16, "f32": jnp.float32}
    lhs, rhs = dt[params.get("lhs", "bf16")], dt[params.get("rhs", "bf16")]

    rhs_t = params.get("rhs_t", False)  # weights stored (n, k) and contracted on their last dim
    dims = (((1,), (1,)), ((), ())) if rhs_t else (((1,), (0,)), ((), ()))

    def kernel(x_hbm, o_hbm, xs, ws, acc):
        def body(_, carry):
            acc[...] += jax.lax.dot_general(xs[...], ws[...], dims, preferred_element_type=jnp.float32)
            return carry

        jax.lax.fori_loop(0, reps, body, 0)

    scratch = [
        pltpu.VMEM((m, kk), lhs),
        pltpu.VMEM((n, kk) if rhs_t else (kk, n), rhs),
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


def sld_chain(params):
    """`n` dependent SMEM loads j = table[j] over a scalar-prefetched cyclic permutation.

    isolates: scalar load-to-use latency.
    """
    n, size = params["n"], params["size"]

    def kernel(tbl_ref, x_hbm, o_ref):
        j = jax.lax.fori_loop(0, n, lambda i, j: tbl_ref[j], 0)
        o_ref[...] = jnp.zeros((8, 128), jnp.float32) + j.astype(jnp.float32)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(1,), in_specs=[HBM],
        out_specs=pl.BlockSpec((8, 128), lambda i, tbl: (0, 0)),
    )

    def fn(tbl, x):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
                              grid_spec=grid_spec, name="k_sld_chain")(tbl, x)

    specs = [jax.ShapeDtypeStruct((size,), jnp.int32), jax.ShapeDtypeStruct((8, 128), jnp.float32)]

    def make_inputs(seed):
        order = np.random.default_rng(seed).permutation(size)
        tbl = np.empty(size, np.int32)
        tbl[order] = np.roll(order, -1)  # one cycle through every entry
        return [tbl, np.zeros((8, 128), np.float32)]

    return fn, specs, make_inputs


def dma_overhead(params):
    """`n` rounds of: start one (8,128) f32 HBM->VMEM copy, run a dependent chain of `k` vadds
    (longer than the copy), wait for the copy. dma=False drops the copy and its wait.

    isolates: core-side cost of a DMA start plus a wait that finds the copy already done.
    """
    n, k, with_dma = params["n"], params["k"], params["dma"]

    def kernel(x_hbm, o_hbm, buf, acc, sem):
        def body(i, carry):
            cp = pltpu.make_async_copy(x_hbm.at[pl.ds(0, 8), :], buf, sem) if with_dma else None
            if with_dma:
                cp.start()
            v = acc[...]
            for _ in range(k):
                v = v + 1.0
            acc[...] = v
            if with_dma:
                cp.wait()
            return carry

        jax.lax.fori_loop(0, n, body, 0)

    scratch = [pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((8, 128), jnp.float32), pltpu.SemaphoreType.DMA(())]
    return _dma_call(kernel, (8192, 128), scratch, "k_dma_overhead")


def unit_chain(params):
    """k rounds of a dependent chain through functional units on one (8, 128) block:
    'mxu' v = v @ W; 'mxu_xlu' adds v -= max(v, lanes); 'mxu_xlu_eup' adds v = exp(...);
    'xlu_eup' drops the matmul. dtype is the matmul LHS/weight dtype (f32 or bf16).

    isolates: result hand-off latency between MXU, XLU and EUP (the chain in decode attention).
    """
    k, kind = params["k"], params["kind"]
    dt = {"bf16": jnp.bfloat16, "f32": jnp.float32}[params.get("dtype", "f32")]

    def kernel(x_hbm, o_hbm, vbuf, wbuf):
        v = vbuf[...]
        w = wbuf[...]
        for _ in range(k):
            if "mxu" in kind:
                v = jnp.dot(v.astype(dt), w, preferred_element_type=jnp.float32)
            if "xlu" in kind:
                m = jnp.max(v, axis=-1, keepdims=True)
                v = jnp.exp(v - m) if "eup" in kind else v - m
        vbuf[...] = v

    scratch = [pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((128, 128), dt)]
    return _no_dma_call(kernel, scratch, "k_unit_chain")


def dma_compute(params):
    """Pipelined stream over a (16384, 1024) f32 array in `block_rows` blocks with `reads` inputs
    (0/1) and `writes` outputs (0/1) and per-step compute: 'none'; 'valu' = `amount` dependent vadds
    on one vreg (no VMEM traffic); 'vmem' = copy an (`amount`, 1024) f32 VMEM scratch block whose
    first rows feed the output (vld/vst traffic, kept live).

    isolates: how core work, and VMEM traffic in particular, interacts with pipelined DMAs.
    """
    rows, cols, br, kind, amount = 16384, 1024, params["block_rows"], params["kind"], params["amount"]
    nr, nw = params.get("reads", 1), params.get("writes", 1)
    blk = pl.BlockSpec((br, cols), lambda i: (i, 0))
    resident = pl.BlockSpec((8, cols), lambda i: (0, 0))

    def kernel(*refs):
        ins, o_ref, s1, s2 = refs[:nr], refs[nr], refs[nr + 1], refs[nr + 2]
        val = ins[0][...] if nr else jnp.full((br, cols), 1.0, jnp.float32)
        if kind == "valu":
            v = s1[pl.ds(0, 8), pl.ds(0, 128)]
            for _ in range(amount):
                v = v + 1.0
            s1[pl.ds(0, 8), pl.ds(0, 128)] = v
        elif kind == "vmem":
            s2[...] = s1[...] + 1.0
            val = val + s2[pl.ds(0, 8), :][0:1, :]
        if nw:
            o_ref[...] = val
        else:
            o_ref[...] = val[:8]

    n = max(amount, 8) if kind == "vmem" else 8

    def fn(*xs):
        return pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct((rows, cols) if nw else (8, cols), jnp.float32),
            grid=(rows // br,), in_specs=[blk] * nr, out_specs=blk if nw else resident,
            scratch_shapes=[pltpu.VMEM((n, cols), jnp.float32), pltpu.VMEM((n, cols), jnp.float32)],
            name="k_dma_compute",
        )(*xs)

    if nr:
        return fn, [jax.ShapeDtypeStruct((rows, cols), jnp.float32)]

    def fn0(dummy):
        return fn()

    return fn0, [jax.ShapeDtypeStruct((8, 128), jnp.float32)]


def sld_use(params):
    """`n` scalar loads table[i] at loop-counter addresses, summed into one scalar.

    isolates: scalar load-to-use latency without an address dependence (contrast sld_chain).
    """
    n, size = params["n"], params["size"]

    def kernel(tbl_ref, x_hbm, o_ref):
        acc = jax.lax.fori_loop(0, n, lambda i, a: a + tbl_ref[i % size], 0)
        o_ref[...] = jnp.zeros((8, 128), jnp.float32) + acc.astype(jnp.float32)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(1,), in_specs=[HBM],
        out_specs=pl.BlockSpec((8, 128), lambda i, tbl: (0, 0)),
    )

    def fn(tbl, x):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
                              grid_spec=grid_spec, name="k_sld_use")(tbl, x)

    specs = [jax.ShapeDtypeStruct((size,), jnp.int32), jax.ShapeDtypeStruct((8, 128), jnp.float32)]

    def make_inputs(seed):
        return [np.arange(size, dtype=np.int32), np.zeros((8, 128), np.float32)]

    return fn, specs, make_inputs


def mxu_fresh(params):
    """reps x (acc += x @ W[i]) with a different (k, n) weight tile per rep, all in VMEM: weights
    are loaded and pushed fresh every rep, as in tiled matmuls (contrast mxu_dot, which reuses W).

    isolates: the VMEM load -> weight push -> matmul path; k < 128 exercises masked pushes.
    """
    m, kk, n, reps = params["m"], params["k"], params["n"], params["reps"]

    def kernel(x_hbm, o_hbm, xs, ws, acc):
        def body(i, carry):
            acc[...] += jnp.dot(xs[...], ws[i], preferred_element_type=jnp.float32)
            return carry

        jax.lax.fori_loop(0, reps, body, 0)

    scratch = [pltpu.VMEM((m, kk), jnp.bfloat16), pltpu.VMEM((reps, kk, n), jnp.bfloat16), pltpu.VMEM((m, n), jnp.float32)]
    return _no_dma_call(kernel, scratch, "k_mxu_fresh")


def vmem_chain(params):
    """n rounds of buf[a] = buf[b] + 1 where a and b come from a scalar-prefetched table and are equal
    at run time, so each round's load must see the previous round's store through VMEM.

    isolates: VMEM store -> load -> use latency (vld latency).
    """
    n = params["n"]

    def kernel(idx_ref, x_hbm, o_ref, buf):
        def body(i, carry):
            buf[pl.ds(idx_ref[2 * i + 1], 8), :] = buf[pl.ds(idx_ref[2 * i], 8), :] + 1.0
            return carry

        jax.lax.fori_loop(0, n, body, 0)
        o_ref[...] = buf[pl.ds(0, 8), :]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(1,), in_specs=[HBM],
        out_specs=pl.BlockSpec((8, 128), lambda i, idx: (0, 0)),
        scratch_shapes=[pltpu.VMEM((64, 128), jnp.float32)],
    )

    def fn(idx, x):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
                              grid_spec=grid_spec, name="k_vmem_chain")(idx, x)

    specs = [jax.ShapeDtypeStruct((2 * n,), jnp.int32), jax.ShapeDtypeStruct((8, 128), jnp.float32)]

    def make_inputs(seed):
        return [np.zeros(2 * n, np.int32), np.zeros((8, 128), np.float32)]

    return fn, specs, make_inputs


def xpose_chain(params):
    """k rounds of v = v.T + 1 on `nv` independent (128, 128) tiles (dtype f32 or bf16), in VMEM.

    isolates: XLU transpose (vxpose start/cont/end + vpop.trf) latency and throughput.
    """
    k, nv = params["k"], params["nv"]
    dt = {"bf16": jnp.bfloat16, "f32": jnp.float32}[params["dtype"]]

    def kernel(x_hbm, o_hbm, buf):
        vs = [buf[i] for i in range(nv)]
        for _ in range(k):
            vs = [(v.T + 1).astype(dt) for v in vs]
        for i, v in enumerate(vs):
            buf[i] = v

    return _no_dma_call(kernel, [pltpu.VMEM((nv, 128, 128), dt)], "k_xpose_chain")


def mxu_masked(params):
    """mxu_fresh with the weight rows at or beyond a run-time bound (scalar prefetch) zeroed, as in
    megablox tgmm group boundaries; aims at masked weight pushes (vmatpush.*.msk)."""
    m, kk, n, reps = params["m"], params["k"], params["n"], params["reps"]

    def kernel(lim_ref, x_hbm, o_ref, xs, ws, acc):
        rows = jax.lax.broadcasted_iota(jnp.int32, (kk, n), 0)

        def body(i, carry):
            w = jnp.where(rows < lim_ref[0], ws[i], jnp.zeros_like(ws[i]))
            acc[...] += jnp.dot(xs[...], w, preferred_element_type=jnp.float32)
            return carry

        jax.lax.fori_loop(0, reps, body, 0)
        o_ref[...] = acc[pl.ds(0, 8), pl.ds(0, 128)]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(1,), in_specs=[HBM],
        out_specs=pl.BlockSpec((8, 128), lambda i, lim: (0, 0)),
        scratch_shapes=[pltpu.VMEM((m, kk), jnp.bfloat16), pltpu.VMEM((reps, kk, n), jnp.bfloat16),
                        pltpu.VMEM((m, n), jnp.float32)],
    )

    def fn(lim, x):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
                              grid_spec=grid_spec, name="k_mxu_masked")(lim, x)

    specs = [jax.ShapeDtypeStruct((1,), jnp.int32), jax.ShapeDtypeStruct((8, 128), jnp.float32)]
    return fn, specs, lambda seed: [np.array([kk // 2], np.int32), np.zeros((8, 128), np.float32)]


def vmem_access(params):
    """`reps` passes over a (rows, 128) f32 VMEM scratch, each pass reading it with the given
    access form and writing the result to a second scratch. form: 'plain' (contiguous rows),
    'stride2' / 'stride4' (sublane-strided reads, every 2nd / 4th row), 'masked' (stores of
    5-row blocks, partial sublane masks), 'stride2_store' (sublane-strided stores).

    isolates: cost of sublane-shuffle loads/stores (vld.sshfl / vst.sshfl) and masked stores
    (vst.msk) relative to plain vld/vst, as used by ragged paged attention.
    """
    rows, reps, form = params["rows"], params["reps"], params["form"]

    def kernel(x_hbm, o_hbm, a, b):
        def body(_, carry):
            if form == "plain":
                b[...] = a[...] + 1.0
            elif form in ("stride2", "stride4"):
                st = 2 if form == "stride2" else 4
                b[pl.ds(0, rows // st), :] = a[pl.ds(0, rows // st, stride=st), :] + 1.0
            elif form == "stride2_store":
                b[pl.ds(0, rows // 2, stride=2), :] = a[pl.ds(0, rows // 2), :] + 1.0
            elif form == "masked":
                for i in range(rows // 8):
                    b[pl.ds(8 * i, 5), :] = a[pl.ds(8 * i, 5), :] + 1.0
            a[pl.ds(0, 8), :] = b[pl.ds(0, 8), :]  # carry a dependence into the next pass
            return carry

        jax.lax.fori_loop(0, reps, body, 0)

    scratch = [pltpu.VMEM((rows, 128), jnp.float32), pltpu.VMEM((rows, 128), jnp.float32)]
    return _no_dma_call(kernel, scratch, "k_vmem_access")


def vmem_ops(params):
    """`reps` passes, each issuing `n` independent copies of one access form from a (16n, 128) f32
    input block into the (8n, 128) f32 output block (both pipelined, so every store stays live):
    'vld' plain copy; 'vst_msk' 5-row partial stores; 'vld_sshfl' sublane-strided loads (every 2nd
    row); 'vst_sshfl' sublane-strided stores; 'vrot_slane' sublane rotate then store.

    isolates: throughput of the access forms ragged paged attention uses, against plain vld/vst.
    """
    n, reps, form = params["n"], params["reps"], params["form"]

    def kernel(a_ref, o_ref):
        def body(p, carry):
            for i in range(n):
                if form == "vld":
                    o_ref[pl.ds(8 * i, 8), :] = a_ref[pl.ds(8 * i, 8), :]
                elif form == "vst_msk":
                    o_ref[pl.ds(8 * i, 5), :] = a_ref[pl.ds(8 * i, 5), :]
                elif form == "vld_sshfl":
                    o_ref[pl.ds(8 * i, 8), :] = a_ref[pl.ds(16 * i, 8, stride=2), :]
                elif form == "vst_sshfl":
                    o_ref[pl.ds(16 * (i // 2) + i % 2, 4, stride=2), :] = a_ref[pl.ds(8 * i, 4), :]
                elif form == "vrot_slane":
                    o_ref[pl.ds(8 * i, 8), :] = pltpu.roll(a_ref[pl.ds(8 * i, 8), :], 1 + (p % 7), 0)
            return carry

        jax.lax.fori_loop(0, reps, body, 0)

    def fn(a):
        return pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct((16 * n, 128), jnp.float32), grid=(1,),
            in_specs=[pl.BlockSpec((16 * n, 128), lambda i: (0, 0))],
            out_specs=pl.BlockSpec((16 * n, 128), lambda i: (0, 0)), name="k_vmem_ops",
        )(a)

    return fn, [jax.ShapeDtypeStruct((16 * n, 128), jnp.float32)]


MICRO = {
    "vmem_ops": (vmem_ops, "micro"),
    "vmem_access": (vmem_access, "micro"),
    "xpose_chain": (xpose_chain, "micro"),
    "mxu_masked": (mxu_masked, "micro"),
    "mxu_fresh": (mxu_fresh, "micro"),
    "vmem_chain": (vmem_chain, "micro"),
    "unit_chain": (unit_chain, "micro"),
    "dma_compute": (dma_compute, "micro"),
    "sld_use": (sld_use, "micro"),
    "sld_chain": (sld_chain, "micro"),
    "dma_overhead": (dma_overhead, "micro"),
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
