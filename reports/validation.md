# Validation

Error = predicted / measured - 1 on the median device time. XLA = `estimated_cycles` from `--xla_jf_dump_static_llo_profile_to` at 1.5 GHz.

## Per-class error, split = valid

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 38 | 2.2% | 4.8% | 12.6% | 9.7% | 18.9% | 19.3% | 35/38 | 37/38 |
| streaming | 9 | 2.6% | 7.7% | 10.5% | 31.4% | 52.2% | 52.3% | 6/9 | 8/9 |
| latency | 16 | 7.5% | 12.1% | 14.1% | - | - | - | 6/16 | 9/16 |

## Per-class error, split = calib

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 40 | 1.8% | 7.3% | 15.4% | 2.1% | 20.1% | 25.8% | 35/40 | 38/40 |
| streaming | 16 | 1.0% | 11.5% | 19.8% | 44.2% | 72.9% | 73.2% | 14/16 | 14/16 |
| latency | 19 | 5.3% | 12.4% | 19.9% | - | - | - | 8/19 | 13/19 |

## Per-class error, supplementary kernels (flash), all splits

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 15 | 1.9% | 3.0% | 3.8% | 4.1% | 31.3% | 35.1% | 15/15 | 15/15 |
| streaming | 0 | - | - | - | - | - | - | 0/0 | 0/0 |
| latency | 0 | - | - | - | - | - | - | 0/0 | 0/0 |

## Ranking within autotune sweeps (all splits)

| sweep | n | Spearman ours | Spearman XLA | top-1 ours | top-1 XLA | top-3 overlap ours | regret ours | regret XLA |
|---|---|---|---|---|---|---|---|---|
| flash{"b": 1, "causal": false, "d": 128, "h": 8, "s": 2048} | 5 | 1.00 | 1.00 | yes | yes | 3/3 | 0.0% | 0.0% |
| flash{"b": 1, "causal": true, "d": 128, "h": 8, "s": 2048} | 5 | 1.00 | 0.90 | yes | no | 3/3 | 0.0% | 2.1% |
| flash{"b": 1, "causal": true, "d": 128, "h": 8, "s": 4096} | 5 | 1.00 | 0.90 | yes | yes | 3/3 | 0.0% | 0.0% |
| gmm{"groups": 16, "k": 2048, "m": 8192, "n": 512} | 4 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| gmm{"groups": 8, "k": 1024, "m": 4096, "n": 1024} | 4 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| layernorm{"cols": 1024, "rows": 8192} | 3 | 0.50 | 1.00 | no | yes | 3/3 | 7.8% | 0.0% |
| matmul_tpu{"k": 1024, "m": 4096, "n": 4096} | 5 | 0.80 | 1.00 | no | yes | 2/3 | 8.2% | 0.0% |
| matmul_tpu{"k": 2048, "m": 2048, "n": 2048} | 5 | 0.90 | 0.90 | no | no | 3/3 | 4.6% | 4.6% |
| matmul{"k": 4096, "m": 4096, "n": 4096} | 3 | 1.00 | 1.00 | yes | yes | 3/3 | 0.0% | 0.0% |
| paged{"batch": 16, "head_dim": 128, "kv_heads": 8, "page_size": 32, "p | 3 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| paged{"batch": 32, "head_dim": 128, "kv_heads": 8, "page_size": 16, "p | 3 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| paged{"batch": 8, "head_dim": 128, "kv_heads": 8, "page_size": 16, "pa | 3 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| prng{"algo": "philox", "cols": 4096, "rows": 4096} | 4 | 0.80 | 0.80 | yes | yes | 2/3 | 0.0% | 0.0% |
| prng{"algo": "threefry", "cols": 4096, "rows": 4096} | 4 | 1.00 | 1.00 | yes | yes | 3/3 | 0.0% | 0.0% |
| rmsnorm{"cols": 2048, "rows": 16384} | 3 | -1.00 | 0.50 | no | no | 3/3 | 5.9% | 0.2% |
| softmax{"cols": 1024, "rows": 8192} | 3 | 0.50 | 1.00 | no | yes | 3/3 | 6.9% | 0.0% |
| splash{"d": 128, "h": 8, "mask": "causal", "mqa": true, "s": 4096} | 5 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| splash{"d": 128, "h": 8, "mask": "causal", "s": 2048, "soft_cap": 30.0 | 5 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| splash{"d": 128, "h": 8, "mask": "causal", "s": 2048, "window": 512} | 5 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| splash{"d": 128, "h": 8, "mask": "causal", "s": 4096, "window": 512} | 5 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| splash{"d": 128, "h": 8, "mask": "causal", "s": 8192} | 5 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| splash{"d": 128, "h": 8, "mask": "chunked", "s": 4096, "window": 1024} | 5 | 0.90 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| splash{"d": 128, "h": 8, "mask": "full", "s": 1024} | 5 | 0.90 | -0.20 | no | no | 3/3 | 1.1% | 1.1% |
| splash{"d": 128, "h": 8, "mask": "full", "s": 2048, "window": 512} | 5 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| splash{"d": 128, "h": 8, "mask": "local", "s": 4096, "window": 512} | 5 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| splash{"d": 256, "h": 4, "mask": "causal", "s": 2048} | 5 | 0.90 | n/a | yes | n/a | 2/3 | 0.0% | n/a |
| splash{"d": 64, "h": 16, "mask": "causal", "s": 4096} | 5 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| tgmm{"groups": 16, "k": 512, "m": 8192, "n": 2048} | 4 | 0.80 | n/a | no | n/a | 3/3 | 9.2% | n/a |
| tgmm{"groups": 8, "k": 1024, "m": 4096, "n": 1024} | 4 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |

## All records

| id | class | split | measured µs | ours | XLA |
|---|---|---|---|---|---|
| add__r2048_c2048_b64x2048 | streaming | calib | 75.51 | -1.1% | -61.6% |
| add__r8192_c2048_b1024x512 | streaming | calib | 295.55 | +0.4% | -73.2% |
| add__r8192_c2048_b256x2048 | streaming | calib | 294.16 | +0.9% | -73.1% |
| col_sum__r16384_c2048_b128 | streaming | calib | 178.03 | +0.9% | -17.8% |
| col_sum__r16384_c2048_b512 | streaming | calib | 178.20 | +0.9% | -39.3% |
| col_sum__r4096_c8192_b64 | streaming | valid | 178.16 | +0.9% | -31.4% |
| flash__s2048_c0_q1024_k512 | compute | calib | 130.94 | +1.1% | +0.3% |
| flash__s2048_c0_q128_k128 | compute | valid | 976.90 | -3.2% | +35.1% |
| flash__s2048_c0_q256_k256 | compute | valid | 312.79 | -1.0% | +24.1% |
| flash__s2048_c0_q512_k1024 | compute | calib | 130.96 | +2.5% | +0.4% |
| flash__s2048_c0_q512_k512 | compute | valid | 137.53 | +1.5% | +3.0% |
| flash__s2048_c1_q1024_k512 | compute | valid | 111.47 | +2.6% | +0.4% |
| flash__s2048_c1_q128_k128 | compute | valid | 663.29 | -2.7% | +32.3% |
| flash__s2048_c1_q256_k256 | compute | calib | 220.99 | +0.9% | +27.2% |
| flash__s2048_c1_q512_k1024 | compute | valid | 114.67 | +1.9% | +2.5% |
| flash__s2048_c1_q512_k512 | compute | valid | 109.14 | +2.6% | +5.8% |
| flash__s4096_c1_q1024_k512 | compute | calib | 363.26 | +2.4% | +0.2% |
| flash__s4096_c1_q128_k128 | compute | calib | 2533.14 | -3.8% | +29.8% |
| flash__s4096_c1_q256_k256 | compute | calib | 810.66 | -0.8% | +23.8% |
| flash__s4096_c1_q512_k1024 | compute | valid | 372.52 | +1.7% | +2.1% |
| flash__s4096_c1_q512_k512 | compute | calib | 370.19 | +1.7% | +4.1% |
| gmm__m4096_k1024_n1024_g8_t128_128_128 | latency | valid | 893.73 | -4.9% | +nan% |
| gmm__m4096_k1024_n1024_g8_t256_512_512 | latency | valid | 128.92 | -8.1% | +nan% |
| gmm__m4096_k1024_n1024_g8_t512_1024_256 | latency | valid | 112.86 | -3.4% | +nan% |
| gmm__m4096_k1024_n1024_g8_t512_512_512 | latency | calib | 121.61 | -3.6% | +nan% |
| gmm__m8192_k2048_n512_g16_t128_128_128 | latency | valid | 1792.61 | -5.5% | +nan% |
| gmm__m8192_k2048_n512_g16_t256_512_512 | latency | calib | 239.66 | -4.0% | +nan% |
| gmm__m8192_k2048_n512_g16_t512_1024_256 | latency | calib | 282.63 | -2.0% | +nan% |
| gmm__m8192_k2048_n512_g16_t512_512_512 | latency | valid | 230.62 | -2.2% | +nan% |
| layernorm__r16384_c4096_b64 | streaming | calib | 816.85 | +1.0% | -49.0% |
| layernorm__r16384_c4096_b8 | streaming | calib | 1285.88 | -19.8% | -0.6% |
| layernorm__r8192_c1024_b256 | streaming | valid | 102.47 | +0.8% | -52.1% |
| layernorm__r8192_c1024_b64 | streaming | valid | 110.46 | -7.0% | -18.5% |
| layernorm__r8192_c1024_b8 | streaming | valid | 366.40 | -0.9% | +44.7% |
| matmul__n1024_256_256_256 | compute | calib | 41.79 | -9.4% | +2.1% |
| matmul__n1024_512_512_1024 | compute | calib | 18.21 | -2.0% | -19.0% |
| matmul__n4096_1024_1024_512 | compute | valid | 755.73 | +0.3% | -1.1% |
| matmul__n4096_256_1024_512 | compute | valid | 1037.78 | -2.5% | -6.2% |
| matmul__n4096_512_512_512 | compute | valid | 956.10 | -4.7% | -14.8% |
| matmul_tpu__2048_2048_2048_1024_1024_256 | compute | calib | 105.57 | +1.1% | -5.0% |
| matmul_tpu__2048_2048_2048_256_256_256 | compute | calib | 306.49 | -7.8% | +7.5% |
| matmul_tpu__2048_2048_2048_256_512_1024 | compute | calib | 156.44 | -3.6% | -20.9% |
| matmul_tpu__2048_2048_2048_512_1024_512 | compute | valid | 110.45 | -5.3% | -9.2% |
| matmul_tpu__2048_2048_2048_512_512_512 | compute | valid | 127.81 | -4.5% | -18.8% |
| matmul_tpu__4096_1024_4096_1024_1024_256 | compute | valid | 223.10 | +2.5% | -9.7% |
| matmul_tpu__4096_1024_4096_256_256_256 | compute | calib | 671.32 | -13.2% | -1.2% |
| matmul_tpu__4096_1024_4096_256_512_1024 | compute | valid | 296.01 | -12.6% | -19.3% |
| matmul_tpu__4096_1024_4096_512_1024_512 | compute | valid | 241.39 | -8.7% | -16.3% |
| matmul_tpu__4096_1024_4096_512_512_512 | compute | calib | 283.77 | -7.3% | -25.8% |
| paged__b16_pps64_pg32_cb16 | latency | calib | 233.86 | -1.7% | +nan% |
| paged__b16_pps64_pg32_cb4 | latency | calib | 659.87 | +6.3% | +nan% |
| paged__b16_pps64_pg32_cb8 | latency | calib | 327.58 | -0.3% | +nan% |
| paged__b32_pps32_pg16_cb16 | latency | calib | 282.85 | +2.8% | +nan% |
| paged__b32_pps32_pg16_cb4 | latency | calib | 703.82 | +5.3% | +nan% |
| paged__b32_pps32_pg16_cb8 | latency | calib | 366.51 | +5.6% | +nan% |
| paged__b8_pps64_pg16_cb16 | latency | valid | 89.62 | +3.3% | +nan% |
| paged__b8_pps64_pg16_cb4 | latency | calib | 222.55 | +5.6% | +nan% |
| paged__b8_pps64_pg16_cb8 | latency | valid | 111.99 | +6.8% | +nan% |
| prng__philox_b1024x1024 | compute | calib | 1082.83 | -15.4% | -15.3% |
| prng__philox_b256x1024 | compute | calib | 914.51 | -0.0% | +0.1% |
| prng__philox_b256x256 | compute | calib | 927.74 | -0.2% | +0.0% |
| prng__philox_b512x512 | compute | calib | 913.18 | -0.0% | +0.2% |
| prng__threefry_b1024x1024 | compute | calib | 338.59 | +0.0% | +0.1% |
| prng__threefry_b256x1024 | compute | valid | 337.25 | -0.1% | +0.2% |
| prng__threefry_b256x256 | compute | calib | 351.23 | -0.6% | -0.1% |
| prng__threefry_b512x512 | compute | calib | 337.79 | -0.1% | +0.2% |
| ragged__decode_kv16_q128 | latency | calib | 2403.73 | -15.6% | +nan% |
| ragged__decode_kv16_q32 | latency | valid | 675.34 | -11.9% | +nan% |
| ragged__decode_kv32_q64 | latency | calib | 805.52 | -10.7% | +nan% |
| ragged__decode_kv8_q16 | latency | calib | 615.84 | -10.8% | +nan% |
| ragged__mixed_kv16_q32 | latency | valid | 316.89 | -12.1% | +nan% |
| ragged__mixed_kv32_q64 | latency | valid | 333.81 | -10.9% | +nan% |
| ragged__mixed_kv8_q16 | latency | valid | 397.32 | -11.4% | +nan% |
| ragged__prefill_kv16_q32 | latency | valid | 348.20 | -12.0% | +nan% |
| ragged__prefill_kv32_q64 | latency | valid | 211.96 | -10.6% | +nan% |
| ragged__prefill_kv8_q16 | latency | calib | 657.39 | -11.6% | +nan% |
| rmsnorm__r16384_c2048_b128 | streaming | calib | 406.87 | +1.4% | -64.8% |
| rmsnorm__r16384_c2048_b256 | streaming | calib | 407.78 | +1.1% | -72.8% |
| rmsnorm__r16384_c2048_b32 | streaming | calib | 430.70 | -4.8% | -29.8% |
| rmsnorm__r8192_c4096_b16 | streaming | valid | 431.87 | -4.4% | -29.3% |
| rmsnorm__r8192_c4096_b64 | streaming | calib | 407.04 | +1.3% | -64.2% |
| rope__t16384_h32_b128 | streaming | calib | 411.30 | +0.7% | -33.8% |
| rope__t8192_h8_b1024 | streaming | calib | 54.45 | +3.2% | -25.8% |
| rope__t8192_h8_b256 | streaming | calib | 56.22 | -0.4% | -21.4% |
| rope__t8192_h8_b64 | streaming | valid | 82.54 | -10.5% | +4.1% |
| softmax__r16384_c4096_b64 | streaming | calib | 818.47 | +0.8% | -53.9% |
| softmax__r16384_c4096_b8 | streaming | calib | 1228.53 | -18.1% | +1.0% |
| softmax__r8192_c1024_b256 | streaming | valid | 102.53 | +0.7% | -52.3% |
| softmax__r8192_c1024_b64 | streaming | valid | 109.64 | -6.7% | -18.7% |
| softmax__r8192_c1024_b8 | streaming | valid | 367.99 | -2.6% | +42.4% |
| splash__h16_s4096_d64_causal_q1024_kv1024_c1024 | compute | calib | 633.59 | -1.5% | +nan% |
| splash__h16_s4096_d64_causal_q128_kv128_c128 | compute | calib | 3768.06 | +0.2% | +nan% |
| splash__h16_s4096_d64_causal_q256_kv512_c256 | compute | calib | 1073.57 | +4.6% | +nan% |
| splash__h16_s4096_d64_causal_q512_kv1024_c512 | compute | calib | 574.93 | +1.8% | +nan% |
| splash__h16_s4096_d64_causal_q512_kv512_c512 | compute | valid | 592.09 | +2.1% | +nan% |
| splash__h4_s2048_d256_causal_q1024_kv1024_c1024 | compute | calib | 85.25 | -2.0% | +nan% |
| splash__h4_s2048_d256_causal_q128_kv128_c128 | compute | calib | 290.25 | +0.7% | +nan% |
| splash__h4_s2048_d256_causal_q256_kv512_c256 | compute | valid | 83.03 | +4.1% | +nan% |
| splash__h4_s2048_d256_causal_q512_kv1024_c512 | compute | valid | 82.23 | -0.4% | +nan% |
| splash__h4_s2048_d256_causal_q512_kv512_c512 | compute | calib | 77.11 | -0.9% | +nan% |
| splash__h8_s1024_d128_full_q1024_kv1024_c1024 | compute | calib | 32.27 | +0.8% | +5.0% |
| splash__h8_s1024_d128_full_q128_kv128_c128 | compute | calib | 219.60 | +0.3% | +nan% |
| splash__h8_s1024_d128_full_q256_kv512_c256 | compute | calib | 61.69 | +5.0% | +nan% |
| splash__h8_s1024_d128_full_q512_kv1024_c512 | compute | valid | 31.92 | +3.3% | +nan% |
| splash__h8_s1024_d128_full_q512_kv512_c512 | compute | valid | 35.36 | +3.0% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q1024_kv1024_c1024 | compute | valid | 97.93 | -0.1% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q128_kv128_c128 | compute | valid | 555.08 | -2.0% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q256_kv512_c256 | compute | valid | 162.86 | +3.3% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q512_kv1024_c512 | compute | valid | 92.72 | +1.2% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q512_kv512_c512 | compute | calib | 87.44 | +3.0% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q1024_kv1024_c1024 | compute | valid | 321.32 | -2.0% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q128_kv128_c128 | compute | calib | 2121.80 | -2.2% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q256_kv512_c256 | compute | valid | 565.88 | +3.3% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q512_kv1024_c512 | compute | calib | 296.38 | +1.8% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q512_kv512_c512 | compute | valid | 309.93 | +1.3% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q1024_kv1024_c1024 | compute | valid | 133.48 | -0.9% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q128_kv128_c128 | compute | calib | 579.66 | -1.8% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q256_kv512_c256 | compute | valid | 196.34 | +3.4% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q512_kv1024_c512 | compute | calib | 131.54 | +1.8% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q512_kv512_c512 | compute | valid | 115.72 | +2.6% | +nan% |
| splash__h8_s8192_d128_causal_q1024_kv1024_c1024 | compute | valid | 1129.26 | -2.1% | +nan% |
| splash__h8_s8192_d128_causal_q128_kv128_c128 | compute | calib | 8355.48 | -2.3% | +nan% |
| splash__h8_s8192_d128_causal_q256_kv512_c256 | compute | calib | 2116.81 | +3.4% | +nan% |
| splash__h8_s8192_d128_causal_q512_kv1024_c512 | compute | valid | 1043.53 | +1.7% | +nan% |
| splash__h8_s8192_d128_causal_q512_kv512_c512 | compute | calib | 1144.87 | +1.2% | +nan% |
| splash__s2048_causal_q1024_kv1024_c512 | compute | calib | 91.41 | +0.8% | +nan% |
| splash__s2048_causal_q128_kv128_c128 | compute | calib | 547.10 | -2.0% | +nan% |
| splash__s2048_causal_q256_kv256_c256 | compute | valid | 181.65 | +2.4% | +nan% |
| splash__s2048_causal_q512_kv1024_c512 | compute | calib | 92.08 | +2.1% | +nan% |
| splash__s2048_causal_q512_kv512_c512 | compute | calib | 89.58 | +1.6% | +nan% |
| splash__s2048_full_q1024_kv1024_c512 | compute | calib | 123.02 | +2.2% | +nan% |
| splash__s2048_full_q128_kv128_c128 | compute | valid | 866.72 | +0.3% | +nan% |
| splash__s2048_full_q256_kv256_c256 | compute | valid | 279.97 | +4.8% | +nan% |
| splash__s2048_full_q512_kv1024_c512 | compute | valid | 119.40 | +2.6% | +nan% |
| splash__s2048_full_q512_kv512_c512 | compute | calib | 130.28 | +2.8% | +nan% |
| splash__s4096_causal_q1024_kv1024_c512 | compute | valid | 292.20 | +0.7% | +nan% |
| splash__s4096_causal_q128_kv128_c128 | compute | valid | 2121.80 | -2.2% | +nan% |
| splash__s4096_causal_q256_kv256_c256 | compute | valid | 676.43 | +2.2% | +nan% |
| splash__s4096_causal_q512_kv1024_c512 | compute | valid | 296.38 | +1.8% | +nan% |
| splash__s4096_causal_q512_kv512_c512 | compute | calib | 309.93 | +1.3% | +nan% |
| splash__s4096_local_q1024_kv1024_c512 | compute | valid | 204.62 | +1.5% | +nan% |
| splash__s4096_local_q128_kv128_c128 | compute | calib | 538.09 | +0.0% | +nan% |
| splash__s4096_local_q256_kv256_c256 | compute | valid | 214.81 | +4.1% | +nan% |
| splash__s4096_local_q512_kv1024_c512 | compute | valid | 169.45 | +1.7% | +nan% |
| splash__s4096_local_q512_kv512_c512 | compute | valid | 134.03 | +1.7% | +nan% |
| tgmm__m4096_k1024_n1024_g8_t128_128_128 | latency | calib | 1061.05 | -11.6% | +nan% |
| tgmm__m4096_k1024_n1024_g8_t256_256_256 | latency | valid | 211.20 | -4.9% | +nan% |
| tgmm__m4096_k1024_n1024_g8_t512_512_1024 | latency | calib | 109.45 | +4.5% | +nan% |
| tgmm__m4096_k1024_n1024_g8_t512_512_512 | latency | valid | 122.48 | -1.1% | +nan% |
| tgmm__m8192_k512_n2048_g16_t128_128_128 | latency | valid | 2379.98 | -14.1% | +nan% |
| tgmm__m8192_k512_n2048_g16_t256_256_256 | latency | calib | 544.21 | -19.9% | +nan% |
| tgmm__m8192_k512_n2048_g16_t512_512_1024 | latency | calib | 249.17 | +3.6% | +nan% |
| tgmm__m8192_k512_n2048_g16_t512_512_512 | latency | calib | 272.02 | -5.3% | +nan% |
