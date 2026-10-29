"""Kernel registry. Each builder maps params to (fn, arg_specs).

fn is an un-jitted callable; arg_specs are jax.ShapeDtypeStruct. The capture
code materializes real inputs for timing, or attaches a topology sharding for
AOT compilation, so builders never touch devices.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _add_one(params):
    rows, cols = params["rows"], params["cols"]
    dtype = jnp.dtype(params.get("dtype", "float32"))

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1

    def fn(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
            name="k_add_one",
        )(x)

    return fn, [jax.ShapeDtypeStruct((rows, cols), dtype)]


def _add(params):
    rows, cols = params["rows"], params["cols"]
    br, bc = params["block_rows"], params["block_cols"]
    dtype = jnp.dtype(params.get("dtype", "float32"))

    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    spec = pl.BlockSpec((br, bc), lambda i, j: (i, j))

    def fn(x, y):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
            grid=(rows // br, cols // bc),
            in_specs=[spec, spec],
            out_specs=spec,
            name="k_add",
        )(x, y)

    arg = jax.ShapeDtypeStruct((rows, cols), dtype)
    return fn, [arg, arg]


def _matmul(params):
    m, n, k = params["m"], params["n"], params["k"]
    bm, bn, bk = params["bm"], params["bn"], params["bk"]
    dtype = jnp.dtype(params.get("dtype", "bfloat16"))
    nk = k // bk

    def kernel(x_ref, y_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        acc_ref[...] += jnp.dot(
            x_ref[...], y_ref[...], preferred_element_type=jnp.float32
        )

        @pl.when(pl.program_id(2) == nk - 1)
        def _():
            o_ref[...] = acc_ref[...].astype(o_ref.dtype)

    def fn(x, y):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), dtype),
            grid=(m // bm, n // bn, nk),
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
                pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary")
            ),
            name="k_matmul",
        )(x, y)

    return fn, [
        jax.ShapeDtypeStruct((m, k), dtype),
        jax.ShapeDtypeStruct((k, n), dtype),
    ]


def _col_sum(params):
    rows, cols, br = params["rows"], params["cols"], params["block_rows"]
    dtype = jnp.dtype(params.get("dtype", "float32"))

    def kernel(x_ref, o_ref):
        @pl.when(pl.program_id(0) == 0)
        def _():
            o_ref[...] = jnp.zeros_like(o_ref)

        o_ref[...] += jnp.sum(x_ref[...], axis=0, keepdims=True)

    def fn(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1, cols), dtype),
            grid=(rows // br,),
            in_specs=[pl.BlockSpec((br, cols), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((1, cols), lambda i: (0, 0)),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
            name="k_col_sum",
        )(x)

    return fn, [jax.ShapeDtypeStruct((rows, cols), dtype)]


KERNELS = {
    "add_one": (_add_one, "trivial"),
    "add": (_add, "streaming"),
    "col_sum": (_col_sum, "streaming"),
    "matmul": (_matmul, "compute"),
}


from bench.micro import MICRO  # noqa: E402

KERNELS.update(MICRO)


def build(name, params):
    builder, kernel_class = KERNELS[name]
    fn, specs = builder(params)
    return fn, specs, kernel_class
