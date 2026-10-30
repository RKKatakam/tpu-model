# Tiling search: model picks vs hardware

Each space was compiled without a TPU (deviceless, `v5e:1x1`), ranked by the cost model, and the model's top picks plus controls spread over its ranking were timed on a v5e. Regret = measured time of the model's pick / best measured time - 1. Per-space details: `reports/search_<space>_check.md`.

| Space | Compiled | Timed | Regret | Spearman | Median abs error |
|---|---|---|---|---|---|
| Dense matmul 4096^3 (bm, bn, bk) | 62 | 12 | +0.7% | 0.97 | 0.8% |
| Splash attention, causal, seq 4096 (block_q, block_kv, block_kv_compute) | 53 | 12 | +0.0% | 0.94 | 2.8% |
| int8 quantized matmul w8a8, 512x4096x4096 (vLLM; batch, out, in blocks) | 48 | 12 | +0.0% | 1.00 | 1.3% |
| int8 quantized matmul w8a16, 512x4096x4096 (vLLM) | 48 | 12 | +0.0% | 1.00 | 0.4% |
| MoE grouped matmul gradient tgmm, 8192x1024x1024, 8 groups (tm, tk, tn) | 48 | 12 | +0.0% | 0.99 | 1.8% |
| Rope, 16k tokens x 32 heads (block_tokens) | 5 | 5 | +0.0% | 1.00 | 10.3% |
| RMSNorm, 16384x4096 (block_rows) | 5 | 5 | +0.4% | 0.70 | 3.4% |

Kernel families also checked by the autotune sweeps in `reports/validation.md` (model picks the measured best in every sweep): ragged paged attention (vLLM), paged attention, flash attention, MoE grouped matmul (gmm, ours and vLLM's), layernorm, softmax.

Notes: RMSNorm's 32/64/128-row tilings are within 0.6% of each other on hardware (a near tie), hence the lower rank correlation. Rope's absolute error is ~10% (streaming DMA after large writes, open), but its ranking is exact. The int8 w8a8 default block sizes measured 2227 us in an earlier session against 51.7 us for the model's pick; a same-session re-check of the default is still to do.
