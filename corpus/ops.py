"""Calibration-corpus kernels from jax.experimental.pallas.ops.tpu plus common hand-written ones.

Builders return (fn, arg_specs) or (fn, arg_specs, make_inputs). make_inputs(seed) returns
concrete numpy arrays, used when an argument must hold valid data (page tables, lengths,
group sizes); the capture records integer arrays so the replay can load them into SMEM.
"""

import importlib

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu import flash_attention as fa
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention_kernel as pa
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

# The megablox package re-exports a `gmm` function that shadows its `gmm` module.
megablox_gmm = importlib.import_module("jax.experimental.pallas.ops.tpu.megablox.gmm")

BF16 = jnp.bfloat16


def _normal(shape, dtype, rng):
    return rng.standard_normal(shape).astype(np.float32).astype(dtype)


def flash(params):
    b, h, s, d = params["b"], params["h"], params["s"], params["d"]
    bs = fa.BlockSizes(
        block_q=params["block_q"], block_k_major=params["block_k_major"],
        block_k=params["block_k"], block_b=params.get("block_b", 1),
    )

    def fn(q, k, v):
        return fa.flash_attention(q, k, v, causal=params["causal"], sm_scale=d ** -0.5, block_sizes=bs)

    spec = jax.ShapeDtypeStruct((b, h, s, d), BF16)
    return fn, [spec, spec, spec]


def _splash_mask(kind, s, h, window):
    if kind == "causal":
        m = sm.CausalMask((s, s))
    elif kind == "full":
        m = sm.FullMask((s, s))
    elif kind == "local":
        m = sm.LocalMask((s, s), window_size=(window, 0), offset=0)
    elif kind == "chunked":
        m = sm.ChunkedCausalMask((s, s), chunk_size=window)
    else:
        raise ValueError(kind)
    return sm.MultiHeadMask([m] * h)


def splash(params):
    h, s, d = params["h"], params["s"], params["d"]
    mqa = params.get("mqa", False)
    bs = sk.BlockSizes(
        block_q=params["block_q"], block_kv=params["block_kv"],
        block_kv_compute=params.get("block_kv_compute", params["block_kv"]),
    )
    make = sk.make_splash_mqa_single_device if mqa else sk.make_splash_mha_single_device
    kernel = make(
        mask=_splash_mask(params["mask"], s, h, params.get("window", 256)), block_sizes=bs,
        attn_logits_soft_cap=params.get("soft_cap"),
    )

    def fn(q, k, v):
        return kernel(q * d ** -0.5, k, v)

    q = jax.ShapeDtypeStruct((h, s, d), BF16)
    kv = jax.ShapeDtypeStruct((s, d) if mqa else (h, s, d), BF16)
    return fn, [q, kv, kv]


def paged(params):
    b, qh, kvh, d = params["batch"], params["q_heads"], params["kv_heads"], params["head_dim"]
    page, pps, total = params["page_size"], params["pages_per_seq"], params["total_pages"]

    def fn(q, k_pages, v_pages, lengths, page_indices):
        return pa.paged_attention(
            q, k_pages, v_pages, lengths, page_indices,
            pages_per_compute_block=params["pages_per_compute_block"],
        )

    specs = [
        jax.ShapeDtypeStruct((b, qh, d), BF16),
        jax.ShapeDtypeStruct((kvh, total, page, d), BF16),
        jax.ShapeDtypeStruct((kvh, total, page, d), BF16),
        jax.ShapeDtypeStruct((b,), jnp.int32),
        jax.ShapeDtypeStruct((b, pps), jnp.int32),
    ]

    def make_inputs(seed):
        rng = np.random.default_rng(seed)
        lo, hi = params.get("min_len", page), pps * page
        lengths = rng.integers(lo, hi + 1, size=b).astype(np.int32)
        page_indices = rng.permutation(total)[: b * pps].reshape(b, pps).astype(np.int32)
        return [
            _normal(specs[0].shape, BF16, rng),
            _normal(specs[1].shape, BF16, rng),
            _normal(specs[2].shape, BF16, rng),
            lengths,
            page_indices,
        ]

    return fn, specs, make_inputs


def gmm(params):
    m, k, n, g = params["m"], params["k"], params["n"], params["groups"]
    tiling = (params["tm"], params["tk"], params["tn"])

    def fn(lhs, rhs, group_sizes):
        return megablox_gmm.gmm(lhs, rhs, group_sizes, preferred_element_type=BF16, tiling=tiling)

    specs = [
        jax.ShapeDtypeStruct((m, k), BF16),
        jax.ShapeDtypeStruct((g, k, n), BF16),
        jax.ShapeDtypeStruct((g,), jnp.int32),
    ]

    def make_inputs(seed):
        rng = np.random.default_rng(seed)
        cuts = np.sort(rng.integers(0, m + 1, size=g - 1))
        sizes = np.diff(np.concatenate([[0], cuts, [m]])).astype(np.int32)
        return [_normal(specs[0].shape, BF16, rng), _normal(specs[1].shape, BF16, rng), sizes]

    return fn, specs, make_inputs


def layernorm(params):
    rows, cols, br = params["rows"], params["cols"], params["block_rows"]
    dtype = jnp.dtype(params.get("dtype", "float32"))

    def kernel(x_ref, g_ref, b_ref, o_ref):
        x = x_ref[...].astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        y = (x - mean) * jax.lax.rsqrt(var + 1e-6) * g_ref[...] + b_ref[...]
        o_ref[...] = y.astype(o_ref.dtype)

    def fn(x, g, b):
        row = pl.BlockSpec((br, cols), lambda i: (i, 0))
        vec = pl.BlockSpec((1, cols), lambda i: (0, 0))
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
            grid=(rows // br,),
            in_specs=[row, vec, vec],
            out_specs=row,
            name="k_layernorm",
        )(x, g, b)

    return fn, [
        jax.ShapeDtypeStruct((rows, cols), dtype),
        jax.ShapeDtypeStruct((1, cols), jnp.float32),
        jax.ShapeDtypeStruct((1, cols), jnp.float32),
    ]


def softmax(params):
    rows, cols, br = params["rows"], params["cols"], params["block_rows"]
    dtype = jnp.dtype(params.get("dtype", "float32"))

    def kernel(x_ref, o_ref):
        x = x_ref[...].astype(jnp.float32)
        e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
        o_ref[...] = (e / jnp.sum(e, axis=-1, keepdims=True)).astype(o_ref.dtype)

    def fn(x):
        spec = pl.BlockSpec((br, cols), lambda i: (i, 0))
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((rows, cols), dtype),
            grid=(rows // br,),
            in_specs=[spec],
            out_specs=spec,
            name="k_softmax",
        )(x)

    return fn, [jax.ShapeDtypeStruct((rows, cols), dtype)]


OPS = {
    "flash": (flash, "compute"),
    "splash": (splash, "compute"),
    "paged": (paged, "latency"),
    "gmm": (gmm, "latency"),
    "layernorm": (layernorm, "streaming"),
    "softmax": (softmax, "streaming"),
}
