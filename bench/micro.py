"""Microbenchmark kernels. Each isolates one mechanism; the bundle check that
confirms it lives in bench/checks.py.

Inputs and outputs stay in HBM (memory_space=HBM) so the pipeline emits no
DMAs; work happens on VMEM scratch whose contents are irrelevant to timing.
"""

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


MICRO = {
    "valu_straight": (valu_straight, "micro"),
    "valu_loop": (valu_loop, "micro"),
    "dma_seq": (dma_seq, "micro"),
}
