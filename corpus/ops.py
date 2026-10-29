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
from jax.experimental.pallas.ops.tpu import matmul as tpu_matmul
from jax.experimental.pallas.ops.tpu.random import philox, threefry
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import kernel as rpa
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


def ragged(params):
    """Mixed prefill + decode batch: `n_decode` sequences with one query token and `kv_len` context,
    `n_prefill` sequences with `prefill_len` query tokens and a `prefix` cached prefix."""
    nd, npf, pl_len, kv_len, prefix = params["n_decode"], params["n_prefill"], params["prefill_len"], params["kv_len"], params["prefix"]
    hq, hkv, d, page = params["q_heads"], params["kv_heads"], params["head_dim"], params["page_size"]
    seqs = nd + npf
    q_lens = [1] * nd + [pl_len] * npf
    kv_lens = [kv_len] * nd + [prefix + pl_len] * npf
    pages_per_seq = -(-max(kv_lens) // page)
    total_pages = max(params.get("total_pages", 0), seqs * pages_per_seq)
    tokens = sum(q_lens)

    def fn(q, kv_pages, kv_lens_a, page_indices, cu_q_lens, num_seqs):
        return rpa.ragged_paged_attention(
            q, kv_pages, kv_lens_a, page_indices, cu_q_lens, num_seqs, sm_scale=d ** -0.5,
            num_kv_pages_per_block=params["kv_pages_per_block"], num_queries_per_block=params["queries_per_block"],
        )

    specs = [
        jax.ShapeDtypeStruct((tokens, hq, d), BF16),
        jax.ShapeDtypeStruct((total_pages, page, 2 * hkv, d), BF16),
        jax.ShapeDtypeStruct((seqs,), jnp.int32),
        jax.ShapeDtypeStruct((seqs, pages_per_seq), jnp.int32),
        jax.ShapeDtypeStruct((seqs + 1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    ]

    def make_inputs(seed):
        rng = np.random.default_rng(seed)
        pages = rng.permutation(total_pages)[: seqs * pages_per_seq].reshape(seqs, pages_per_seq)
        return [
            _normal(specs[0].shape, BF16, rng),
            _normal(specs[1].shape, BF16, rng),
            np.array(kv_lens, np.int32),
            pages.astype(np.int32),
            np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
            np.array([seqs], np.int32),
        ]

    return fn, specs, make_inputs


def matmul_tpu(params):
    m, k, n = params["m"], params["k"], params["n"]

    def fn(x, y):
        return tpu_matmul.matmul(x, y, block_shape=(params["bm"], params["bn"]), block_k=params["bk"])

    return fn, [jax.ShapeDtypeStruct((m, k), BF16), jax.ShapeDtypeStruct((k, n), BF16)]


def tgmm(params):
    m, k, n, g = params["m"], params["k"], params["n"], params["groups"]

    def fn(lhs, rhs, group_sizes):
        return megablox_gmm.tgmm(lhs, rhs, group_sizes, preferred_element_type=jnp.float32,
                                 tiling=(params["tm"], params["tk"], params["tn"]))

    specs = [jax.ShapeDtypeStruct((k, m), BF16), jax.ShapeDtypeStruct((m, n), BF16),
             jax.ShapeDtypeStruct((g,), jnp.int32)]

    def make_inputs(seed):
        rng = np.random.default_rng(seed)
        cuts = np.sort(rng.integers(0, m + 1, size=g - 1))
        sizes = np.diff(np.concatenate([[0], cuts, [m]])).astype(np.int32)
        return [_normal(specs[0].shape, BF16, rng), _normal(specs[1].shape, BF16, rng), sizes]

    return fn, specs, make_inputs


def prng(params):
    shape, block = (params["rows"], params["cols"]), (params["block_rows"], params["block_cols"])
    gen = {"threefry": threefry.threefry_2x32_count, "philox": philox.philox_4x32_kernel}[params["algo"]]

    def fn(key):
        return gen(key, shape, shape, block)

    return fn, [jax.ShapeDtypeStruct((2,), jnp.uint32)], lambda seed: [np.array([seed, 42], np.uint32)]


def rmsnorm(params):
    """y = rmsnorm(x + residual) * gamma, also returning x + residual (the transformer pre-norm)."""
    rows, cols, br = params["rows"], params["cols"], params["block_rows"]

    def kernel(x_ref, r_ref, g_ref, y_ref, h_ref):
        h = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        y = h * jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + 1e-6) * g_ref[...]
        y_ref[...] = y.astype(y_ref.dtype)
        h_ref[...] = h.astype(h_ref.dtype)

    def fn(x, r, g):
        row = pl.BlockSpec((br, cols), lambda i: (i, 0))
        vec = pl.BlockSpec((1, cols), lambda i: (0, 0))
        return pl.pallas_call(
            kernel, out_shape=[jax.ShapeDtypeStruct((rows, cols), BF16)] * 2, grid=(rows // br,),
            in_specs=[row, row, vec], out_specs=[row, row], name="k_rmsnorm",
        )(x, r, g)

    return fn, [jax.ShapeDtypeStruct((rows, cols), BF16), jax.ShapeDtypeStruct((rows, cols), BF16),
                jax.ShapeDtypeStruct((1, cols), jnp.float32)]


def rope(params):
    """Rotary embedding on (tokens, heads * head_dim) bf16 with per-token cos/sin tables."""
    tokens, heads, d, bt = params["tokens"], params["heads"], params["head_dim"], params["block_tokens"]
    half = d // 2

    def kernel(x_ref, cos_ref, sin_ref, o_ref):
        cos, sin = cos_ref[...], sin_ref[...]
        for h in range(heads):
            x = x_ref[:, pl.ds(h * d, d)].astype(jnp.float32)
            x1, x2 = x[:, :half], x[:, half:]
            o_ref[:, pl.ds(h * d, d)] = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(o_ref.dtype)

    def fn(x, cos, sin):
        xs = pl.BlockSpec((bt, heads * d), lambda i: (i, 0))
        cs = pl.BlockSpec((bt, half), lambda i: (i, 0))
        return pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct((tokens, heads * d), BF16), grid=(tokens // bt,),
            in_specs=[xs, cs, cs], out_specs=xs, name="k_rope",
        )(x, cos, sin)

    return fn, [jax.ShapeDtypeStruct((tokens, heads * d), BF16), jax.ShapeDtypeStruct((tokens, half), jnp.float32),
                jax.ShapeDtypeStruct((tokens, half), jnp.float32)]


OPS = {
    "flash": (flash, "compute"),
    "splash": (splash, "compute"),
    "paged": (paged, "latency"),
    "gmm": (gmm, "latency"),
    "layernorm": (layernorm, "streaming"),
    "softmax": (softmax, "streaming"),
    "ragged": (ragged, "latency"),
    "matmul_tpu": (matmul_tpu, "compute"),
    "tgmm": (tgmm, "latency"),
    "prng": (prng, "compute"),
    "rmsnorm": (rmsnorm, "streaming"),
    "rope": (rope, "streaming"),
}
