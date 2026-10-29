# External test set (vllm-project/tpu-inference kernels)

Never used to fit the model. Error = predicted / measured - 1 on median device time (sum over the program's Pallas kernels). XLA = static estimate where its profiler accepts the kernel.

| kernel | n | ours median abs | ours worst | within 2% | within 5% | within 10% | XLA median abs (n) |
|---|---|---|---|---|---|---|---|
| vllm_flash | 3 | 2.0% | 2.2% | 2 | 3 | 3 | 3.3% (3) |
| vllm_gmm | 2 | 4.0% | 6.2% | 1 | 1 | 2 | n/a |
| vllm_qmm | 4 | 12.8% | 19.7% | 0 | 1 | 2 | 38.3% (4) |
| vllm_rpa | 9 | 0.7% | 1.9% | 9 | 9 | 9 | n/a |
| **all** | 18 | 1.5% | 19.7% | 12 | 14 | 16 | |

## Ranking within sweeps

| sweep | n | Spearman ours | top-1 ours | regret ours |
|---|---|---|---|---|

## All records

| id | kernels | measured us | ours | XLA |
|---|---|---|---|---|
| vllm_flash__s2048_c1_q256_k256 | 1 | 221.15 | +2.2% | +27.1% |
| vllm_flash__s2048_c1_q512_k1024 | 1 | 114.69 | +2.0% | +2.5% |
| vllm_flash__s4096_c0_q512_k512 | 1 | 536.14 | +1.4% | +3.3% |
| vllm_gmm__m4096_k2048_n1024_g8_t128_128_128 | 1 | 1769.48 | -6.2% | +nan% |
| vllm_gmm__m4096_k2048_n1024_g8_t512_1024_512 | 1 | 205.71 | -1.7% | +nan% |
| vllm_qmm__w8a16_m16_k4096_n4096 | 1 | 308.19 | +6.2% | +64.6% |
| vllm_qmm__w8a16_m512_k4096_n4096 | 1 | 1331.49 | +2.6% | +56.7% |
| vllm_qmm__w8a8_m16_k4096_n4096 | 1 | 525.90 | -19.7% | +19.9% |
| vllm_qmm__w8a8_m512_k4096_n4096 | 1 | 2227.28 | -19.4% | +16.0% |
| vllm_rpa__decode_m128_512 | 2 | 357.39 | +0.7% | +nan% |
| vllm_rpa__decode_m32_128 | 2 | 357.41 | +0.7% | +nan% |
| vllm_rpa__decode_m64_256 | 2 | 357.37 | +0.8% | +nan% |
| vllm_rpa__mixed_m128_512 | 2 | 158.21 | +0.4% | +nan% |
| vllm_rpa__mixed_m32_128 | 2 | 198.91 | -1.0% | +nan% |
| vllm_rpa__mixed_m64_256 | 2 | 182.04 | +0.7% | +nan% |
| vllm_rpa__prefill_m128_512 | 2 | 151.91 | +0.0% | +nan% |
| vllm_rpa__prefill_m32_128 | 2 | 216.98 | -1.9% | +nan% |
| vllm_rpa__prefill_m64_256 | 2 | 187.14 | +0.4% | +nan% |
