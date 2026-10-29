# External test set (vllm-project/tpu-inference kernels)

Never used to fit the model. Error = predicted / measured - 1 on median device time (sum over the program's Pallas kernels). XLA = static estimate where its profiler accepts the kernel.

| kernel | n | ours median abs | ours worst | within 2% | within 5% | within 10% | XLA median abs (n) |
|---|---|---|---|---|---|---|---|
| vllm_flash | 6 | 2.1% | 2.7% | 2 | 6 | 6 | 4.5% (6) |
| vllm_gmm | 3 | 2.3% | 6.2% | 1 | 2 | 3 | n/a |
| vllm_qmm | 8 | 11.8% | 18.7% | 1 | 3 | 4 | 35.0% (8) |
| vllm_rpa | 9 | 0.7% | 1.8% | 9 | 9 | 9 | n/a |
| **all** | 26 | 1.9% | 18.7% | 13 | 20 | 22 | |

## Ranking within sweeps

| sweep | n | Spearman ours | top-1 ours | regret ours |
|---|---|---|---|---|
| vllm_flash{"b": 1, "causal": false, "d": 128, "h": 8, "s": 4096} | 3 | 1.00 | yes | 0.0% |
| vllm_flash{"b": 1, "causal": true, "d": 128, "h": 8, "s": 2048} | 3 | 1.00 | yes | 0.0% |
| vllm_gmm{"groups": 8, "k": 2048, "m": 4096, "n": 1024} | 3 | 1.00 | yes | 0.0% |
| vllm_rpa{"head_dim": 128, "kv_heads": 8, "kv_len": 0, "n_decode": 0, "n_prefill" | 3 | 1.00 | yes | 0.0% |
| vllm_rpa{"head_dim": 128, "kv_heads": 8, "kv_len": 1024, "n_decode": 16, "n_pref | 3 | 1.00 | yes | 0.0% |
| vllm_rpa{"head_dim": 128, "kv_heads": 8, "kv_len": 2048, "n_decode": 32, "n_pref | 3 | -0.50 | no | 0.0% |

## All records

| id | kernels | measured us | ours | XLA |
|---|---|---|---|---|
| vllm_flash__s2048_c1_q256_k256 | 1 | 221.15 | +2.2% | +27.1% |
| vllm_flash__s2048_c1_q512_k1024 | 1 | 114.69 | +2.0% | +2.5% |
| vllm_flash__s2048_c1_q512_k512 | 1 | 109.39 | +2.7% | +5.6% |
| vllm_flash__s4096_c0_q256_k256 | 1 | 1229.87 | +0.1% | +25.3% |
| vllm_flash__s4096_c0_q512_k1024 | 1 | 509.05 | +2.5% | +0.3% |
| vllm_flash__s4096_c0_q512_k512 | 1 | 536.14 | +1.4% | +3.3% |
| vllm_gmm__m4096_k2048_n1024_g8_t128_128_128 | 1 | 1769.48 | -6.2% | +nan% |
| vllm_gmm__m4096_k2048_n1024_g8_t256_512_512 | 1 | 238.57 | -2.3% | +nan% |
| vllm_gmm__m4096_k2048_n1024_g8_t512_1024_512 | 1 | 205.71 | -1.7% | +nan% |
| vllm_qmm__w8a16_m128_k4096_n4096 | 1 | 344.90 | -7.0% | +50.0% |
| vllm_qmm__w8a16_m16_k4096_n4096 | 1 | 308.19 | -1.0% | +64.6% |
| vllm_qmm__w8a16_m2048_k4096_n4096 | 1 | 5302.07 | -5.0% | +55.9% |
| vllm_qmm__w8a16_m512_k4096_n4096 | 1 | 1331.49 | -4.0% | +56.7% |
| vllm_qmm__w8a8_m128_k4096_n4096 | 1 | 533.60 | -18.7% | +20.0% |
| vllm_qmm__w8a8_m16_k4096_n4096 | 1 | 525.90 | -17.1% | +19.9% |
| vllm_qmm__w8a8_m2048_k4096_n4096 | 1 | 8674.20 | -16.7% | +18.1% |
| vllm_qmm__w8a8_m512_k4096_n4096 | 1 | 2227.28 | -16.9% | +16.0% |
| vllm_rpa__decode_m128_512 | 2 | 357.39 | +0.8% | +nan% |
| vllm_rpa__decode_m32_128 | 2 | 357.41 | +0.7% | +nan% |
| vllm_rpa__decode_m64_256 | 2 | 357.37 | +0.8% | +nan% |
| vllm_rpa__mixed_m128_512 | 2 | 158.21 | +0.4% | +nan% |
| vllm_rpa__mixed_m32_128 | 2 | 198.91 | -1.0% | +nan% |
| vllm_rpa__mixed_m64_256 | 2 | 182.04 | +0.7% | +nan% |
| vllm_rpa__prefill_m128_512 | 2 | 151.91 | +0.0% | +nan% |
| vllm_rpa__prefill_m32_128 | 2 | 216.98 | -1.8% | +nan% |
| vllm_rpa__prefill_m64_256 | 2 | 187.14 | +0.4% | +nan% |
