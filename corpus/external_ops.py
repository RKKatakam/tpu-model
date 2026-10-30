"""Builders for kernels from vllm-project/tpu-inference (vendored in corpus/external, see SOURCE).

External test set only: these records are reported separately (scripts/report_external.py) and are
never used to calibrate the model. Inputs follow the repository's own kernel tests.
"""

import numpy as np

import jax
import jax.numpy as jnp

from corpus.external import load_tpu_inference

load_tpu_inference()

from tpu_inference.kernels.flash_attention import kernel as vfa  # noqa: E402
from tpu_inference.kernels.megablox import gmm as vgmm  # noqa: E402
from tpu_inference.kernels.quantized_matmul import kernel as vqmm  # noqa: E402
from tpu_inference.kernels.ragged_paged_attention.v3 import kernel as vrpa  # noqa: E402

BF16 = jnp.bfloat16


def _normal(shape, dtype, rng):
    return rng.standard_normal(shape).astype(np.float32).astype(dtype)


def vllm_rpa(params):
    """Ragged paged attention v3 (attention only, update_kv_cache=False). Decode sequences (one query
    token, `kv_len` context) take the decode kernel; prefill sequences (`prefill_len` tokens after a
    `prefix` cached prefix) take the mixed kernel. m_block_sizes = (bq, bkv, bq_c, bkv_c)."""
    nd, npf, pl_len, kv_len, prefix = (params[k] for k in ("n_decode", "n_prefill", "prefill_len", "kv_len", "prefix"))
    hq, hkv, d, page = params["q_heads"], params["kv_heads"], params["head_dim"], params["page_size"]
    q_lens = [1] * nd + [pl_len] * npf
    kv_lens = [kv_len] * nd + [prefix + pl_len] * npf
    seqs = len(q_lens)
    max_seq = max(8, -(-seqs // 8) * 8)
    tokens = max(128, -(-sum(q_lens) // 128) * 128)
    pps = -(-max(kv_lens) // page)
    num_pages = max(params.get("num_pages", 0), seqs * pps + 1)
    kv_shape = vrpa.get_kv_cache_shape(num_pages, page, hkv, d, BF16)

    def fn(q, k, v, kv_cache, kv_lens_a, page_indices, cu_q_lens, distribution):
        out, _ = vrpa.ragged_paged_attention(
            q, k, v, kv_cache, kv_lens_a, page_indices, cu_q_lens, distribution,
            update_kv_cache=False, sm_scale=d ** -0.5, m_block_sizes=tuple(params["m_blocks"]),
            d_block_sizes=tuple(params["d_blocks"]) if params.get("d_blocks") else None,
            vmem_limit_bytes=params.get("vmem_limit_bytes"),
        )
        return out

    specs = [
        jax.ShapeDtypeStruct((tokens, hq, d), BF16), jax.ShapeDtypeStruct((tokens, hkv, d), BF16),
        jax.ShapeDtypeStruct((tokens, hkv, d), BF16), jax.ShapeDtypeStruct(kv_shape, BF16),
        jax.ShapeDtypeStruct((max_seq,), jnp.int32), jax.ShapeDtypeStruct((max_seq * pps,), jnp.int32),
        jax.ShapeDtypeStruct((max_seq + 1,), jnp.int32), jax.ShapeDtypeStruct((3,), jnp.int32),
    ]

    def make_inputs(seed):
        rng = np.random.default_rng(seed)
        pages = rng.permutation(num_pages)[: max_seq * pps].astype(np.int32)
        cu = np.zeros(max_seq + 1, np.int32)
        cu[1: seqs + 1] = np.cumsum(q_lens)
        cu[seqs + 1:] = cu[seqs]
        lens = np.zeros(max_seq, np.int32)
        lens[:seqs] = kv_lens
        return [
            _normal(specs[0].shape, BF16, rng), _normal(specs[1].shape, BF16, rng), _normal(specs[2].shape, BF16, rng),
            _normal(kv_shape, BF16, rng), lens, pages, cu, np.array([nd, nd, seqs], np.int32),
        ]

    return fn, specs, make_inputs


def vllm_qmm(params):
    """Per-channel int8 weight quantized matmul: x (m, k) bf16 @ w_q (n, k) int8 * w_scale (n,).
    w8a8 quantizes x to int8 inside the kernel; w8a16 keeps bf16 activations."""
    m, k, n = params["m"], params["k"], params["n"]
    x_q = jnp.int8 if params["mode"] == "w8a8" else BF16

    tuned = None
    if params.get("blocks"):  # (batch, out, in) block sizes instead of the kernel's tuned table
        from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import TunedValue
        tuned = TunedValue(*params["blocks"])

    def fn(x, w_q, w_scale):
        return vqmm.quantized_matmul_kernel(x, w_q, w_scale, x_q_dtype=x_q, tuned_value=tuned)

    specs = [jax.ShapeDtypeStruct((m, k), BF16), jax.ShapeDtypeStruct((n, k), jnp.int8),
             jax.ShapeDtypeStruct((n,), jnp.float32)]

    def make_inputs(seed):
        rng = np.random.default_rng(seed)
        return [_normal((m, k), BF16, rng), rng.integers(-127, 128, size=(n, k)).astype(np.int8),
                (rng.random(n) * 0.01 + 1e-3).astype(np.float32)]

    return fn, specs, make_inputs


def vllm_flash(params):
    b, h, s, d = params["b"], params["h"], params["s"], params["d"]
    bs = vfa.BlockSizes(block_q=params["block_q"], block_k_major=params["block_k_major"],
                        block_k=params["block_k"], block_b=1)

    def fn(q, k, v):
        return vfa.flash_attention(q, k, v, causal=params["causal"], sm_scale=d ** -0.5, block_sizes=bs)

    spec = jax.ShapeDtypeStruct((b, h, s, d), BF16)
    return fn, [spec, spec, spec]


def vllm_gmm(params):
    m, k, n, g = params["m"], params["k"], params["n"], params["groups"]

    def fn(lhs, rhs, group_sizes):
        return vgmm.gmm(lhs, rhs, group_sizes, preferred_element_type=BF16, tiling=(params["tm"], params["tk"], params["tn"]))

    specs = [jax.ShapeDtypeStruct((m, k), BF16), jax.ShapeDtypeStruct((g, k, n), BF16), jax.ShapeDtypeStruct((g,), jnp.int32)]

    def make_inputs(seed):
        rng = np.random.default_rng(seed)
        cuts = np.sort(rng.integers(0, m + 1, size=g - 1))
        sizes = np.diff(np.concatenate([[0], cuts, [m]])).astype(np.int32)
        return [_normal(specs[0].shape, BF16, rng), _normal(specs[1].shape, BF16, rng), sizes]

    return fn, specs, make_inputs


EXTERNAL = {
    "vllm_rpa": (vllm_rpa, "external_latency"),
    "vllm_qmm": (vllm_qmm, "external_compute"),
    "vllm_flash": (vllm_flash, "external_compute"),
    "vllm_gmm": (vllm_gmm, "external_latency"),
}
