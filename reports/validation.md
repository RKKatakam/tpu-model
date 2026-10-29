# Validation

Error = predicted / measured - 1 on the median device time. XLA = `estimated_cycles` from `--xla_jf_dump_static_llo_profile_to` at 1.5 GHz.

## Per-class error, split = valid

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 32 | 3.7% | 6.0% | 7.5% | 6.2% | 13.1% | 14.8% | 26/32 | 32/32 |
| streaming | 7 | 3.6% | 4.9% | 5.1% | 42.4% | 52.2% | 52.3% | 6/7 | 7/7 |
| latency | 7 | 4.9% | 8.0% | 8.2% | - | - | - | 4/7 | 7/7 |

## Per-class error, split = calib

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 28 | 2.8% | 6.4% | 9.3% | 5.0% | 16.2% | 19.0% | 24/28 | 28/28 |
| streaming | 9 | 3.1% | 18.0% | 19.4% | 49.0% | 73.1% | 73.2% | 7/9 | 7/9 |
| latency | 10 | 3.9% | 7.0% | 7.6% | - | - | - | 6/10 | 10/10 |

## Per-class error, supplementary kernels (flash), all splits

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 15 | 3.1% | 4.2% | 4.4% | 4.1% | 31.3% | 35.1% | 15/15 | 15/15 |
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
| matmul{"k": 4096, "m": 4096, "n": 4096} | 3 | 1.00 | 1.00 | yes | yes | 3/3 | 0.0% | 0.0% |
| paged{"batch": 16, "head_dim": 128, "kv_heads": 8, "page_size": 32, "p | 3 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| paged{"batch": 32, "head_dim": 128, "kv_heads": 8, "page_size": 16, "p | 3 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| paged{"batch": 8, "head_dim": 128, "kv_heads": 8, "page_size": 16, "pa | 3 | 1.00 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
| softmax{"cols": 1024, "rows": 8192} | 3 | 0.50 | 1.00 | no | yes | 3/3 | 6.9% | 0.0% |
| splash{"d": 128, "h": 8, "mask": "causal", "mqa": true, "s": 4096} | 5 | 0.90 | n/a | yes | n/a | 3/3 | 0.0% | n/a |
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

## All records

| id | class | split | measured µs | ours | XLA |
|---|---|---|---|---|---|
| add__r2048_c2048_b64x2048 | streaming | calib | 75.51 | +0.9% | -61.6% |
| add__r8192_c2048_b1024x512 | streaming | calib | 295.55 | +2.6% | -73.2% |
| add__r8192_c2048_b256x2048 | streaming | calib | 294.16 | +3.1% | -73.1% |
| col_sum__r16384_c2048_b128 | streaming | calib | 178.03 | +1.2% | -17.8% |
| col_sum__r16384_c2048_b512 | streaming | calib | 178.20 | +1.2% | -39.3% |
| col_sum__r4096_c8192_b64 | streaming | valid | 178.16 | +1.2% | -31.4% |
| flash__s2048_c0_q1024_k512 | compute | calib | 130.94 | +2.1% | +0.3% |
| flash__s2048_c0_q128_k128 | compute | valid | 976.90 | -1.7% | +35.1% |
| flash__s2048_c0_q256_k256 | compute | valid | 312.79 | +1.4% | +24.1% |
| flash__s2048_c0_q512_k1024 | compute | calib | 130.96 | +3.9% | +0.4% |
| flash__s2048_c0_q512_k512 | compute | valid | 137.53 | +3.1% | +3.0% |
| flash__s2048_c1_q1024_k512 | compute | valid | 111.47 | +4.2% | +0.4% |
| flash__s2048_c1_q128_k128 | compute | valid | 663.29 | -1.5% | +32.3% |
| flash__s2048_c1_q256_k256 | compute | calib | 220.99 | +2.9% | +27.2% |
| flash__s2048_c1_q512_k1024 | compute | valid | 114.67 | +3.3% | +2.5% |
| flash__s2048_c1_q512_k512 | compute | valid | 109.14 | +4.4% | +5.8% |
| flash__s4096_c1_q1024_k512 | compute | calib | 363.26 | +4.0% | +0.2% |
| flash__s4096_c1_q128_k128 | compute | calib | 2533.14 | -2.6% | +29.8% |
| flash__s4096_c1_q256_k256 | compute | calib | 810.66 | +1.2% | +23.8% |
| flash__s4096_c1_q512_k1024 | compute | valid | 372.52 | +3.1% | +2.1% |
| flash__s4096_c1_q512_k512 | compute | calib | 370.19 | +3.5% | +4.1% |
| gmm__m4096_k1024_n1024_g8_t128_128_128 | latency | valid | 893.73 | -4.9% | +nan% |
| gmm__m4096_k1024_n1024_g8_t256_512_512 | latency | valid | 128.92 | -7.9% | +nan% |
| gmm__m4096_k1024_n1024_g8_t512_1024_256 | latency | valid | 112.86 | -3.1% | +nan% |
| gmm__m4096_k1024_n1024_g8_t512_512_512 | latency | calib | 121.61 | -3.0% | +nan% |
| gmm__m8192_k2048_n512_g16_t128_128_128 | latency | valid | 1792.61 | -5.5% | +nan% |
| gmm__m8192_k2048_n512_g16_t256_512_512 | latency | calib | 239.66 | -3.8% | +nan% |
| gmm__m8192_k2048_n512_g16_t512_1024_256 | latency | calib | 282.63 | -1.5% | +nan% |
| gmm__m8192_k2048_n512_g16_t512_512_512 | latency | valid | 230.62 | -1.8% | +nan% |
| layernorm__r16384_c4096_b64 | streaming | calib | 816.85 | +4.0% | -49.0% |
| layernorm__r16384_c4096_b8 | streaming | calib | 1285.88 | -19.4% | -0.6% |
| layernorm__r8192_c1024_b256 | streaming | valid | 102.47 | +3.7% | -52.1% |
| layernorm__r8192_c1024_b64 | streaming | valid | 110.46 | -5.1% | -18.5% |
| layernorm__r8192_c1024_b8 | streaming | valid | 366.40 | +0.0% | +44.7% |
| matmul__n1024_256_256_256 | compute | calib | 41.79 | -9.3% | +2.1% |
| matmul__n1024_512_512_1024 | compute | calib | 18.21 | -1.4% | -19.0% |
| matmul__n4096_1024_1024_512 | compute | valid | 755.73 | +0.7% | -1.1% |
| matmul__n4096_256_1024_512 | compute | valid | 1037.78 | -2.2% | -6.2% |
| matmul__n4096_512_512_512 | compute | valid | 956.10 | -4.4% | -14.8% |
| paged__b16_pps64_pg32_cb16 | latency | calib | 233.86 | -0.7% | +nan% |
| paged__b16_pps64_pg32_cb4 | latency | calib | 659.87 | +7.6% | +nan% |
| paged__b16_pps64_pg32_cb8 | latency | calib | 327.58 | +1.0% | +nan% |
| paged__b32_pps32_pg16_cb16 | latency | calib | 282.85 | +3.9% | +nan% |
| paged__b32_pps32_pg16_cb4 | latency | calib | 703.82 | +6.6% | +nan% |
| paged__b32_pps32_pg16_cb8 | latency | calib | 366.51 | +6.9% | +nan% |
| paged__b8_pps64_pg16_cb16 | latency | valid | 89.62 | +4.4% | +nan% |
| paged__b8_pps64_pg16_cb4 | latency | calib | 222.55 | +6.9% | +nan% |
| paged__b8_pps64_pg16_cb8 | latency | valid | 111.99 | +8.2% | +nan% |
| softmax__r16384_c4096_b64 | streaming | calib | 818.47 | +3.8% | -53.9% |
| softmax__r16384_c4096_b8 | streaming | calib | 1228.53 | -17.7% | +1.0% |
| softmax__r8192_c1024_b256 | streaming | valid | 102.53 | +3.6% | -52.3% |
| softmax__r8192_c1024_b64 | streaming | valid | 109.64 | -4.8% | -18.7% |
| softmax__r8192_c1024_b8 | streaming | valid | 367.99 | -1.9% | +42.4% |
| splash__h16_s4096_d64_causal_q1024_kv1024_c1024 | compute | calib | 633.59 | -1.0% | +nan% |
| splash__h16_s4096_d64_causal_q128_kv128_c128 | compute | calib | 3768.06 | +1.9% | +nan% |
| splash__h16_s4096_d64_causal_q256_kv512_c256 | compute | calib | 1073.57 | +7.4% | +nan% |
| splash__h16_s4096_d64_causal_q512_kv1024_c512 | compute | calib | 574.93 | +4.1% | +nan% |
| splash__h16_s4096_d64_causal_q512_kv512_c512 | compute | valid | 592.09 | +3.9% | +nan% |
| splash__h4_s2048_d256_causal_q1024_kv1024_c1024 | compute | calib | 85.25 | -1.7% | +nan% |
| splash__h4_s2048_d256_causal_q128_kv128_c128 | compute | calib | 290.25 | +2.6% | +nan% |
| splash__h4_s2048_d256_causal_q256_kv512_c256 | compute | valid | 83.03 | +6.9% | +nan% |
| splash__h4_s2048_d256_causal_q512_kv1024_c512 | compute | valid | 82.23 | +0.6% | +nan% |
| splash__h4_s2048_d256_causal_q512_kv512_c512 | compute | calib | 77.11 | -0.3% | +nan% |
| splash__h8_s1024_d128_full_q1024_kv1024_c1024 | compute | calib | 32.27 | +1.1% | +5.0% |
| splash__h8_s1024_d128_full_q128_kv128_c128 | compute | calib | 219.60 | +1.9% | +nan% |
| splash__h8_s1024_d128_full_q256_kv512_c256 | compute | calib | 61.69 | +7.6% | +nan% |
| splash__h8_s1024_d128_full_q512_kv1024_c512 | compute | valid | 31.92 | +5.0% | +nan% |
| splash__h8_s1024_d128_full_q512_kv512_c512 | compute | valid | 35.36 | +4.4% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q1024_kv1024_c1024 | compute | valid | 97.93 | +0.8% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q128_kv128_c128 | compute | valid | 555.08 | -0.6% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q256_kv512_c256 | compute | valid | 162.86 | +5.8% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q512_kv1024_c512 | compute | valid | 92.72 | +3.3% | +nan% |
| splash__h8_s2048_d128_causal_soft_cap30_q512_kv512_c512 | compute | calib | 87.44 | +5.0% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q1024_kv1024_c1024 | compute | valid | 321.32 | -1.5% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q128_kv128_c128 | compute | calib | 2121.80 | -0.8% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q256_kv512_c256 | compute | valid | 565.88 | +6.0% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q512_kv1024_c512 | compute | calib | 296.38 | +3.8% | +nan% |
| splash__h8_s4096_d128_causal_mqa_q512_kv512_c512 | compute | valid | 309.93 | +3.0% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q1024_kv1024_c1024 | compute | valid | 133.48 | +0.0% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q128_kv128_c128 | compute | calib | 579.66 | -0.4% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q256_kv512_c256 | compute | valid | 196.34 | +6.0% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q512_kv1024_c512 | compute | calib | 131.54 | +3.6% | +nan% |
| splash__h8_s4096_d128_chunked_window1024_q512_kv512_c512 | compute | valid | 115.72 | +4.4% | +nan% |
| splash__h8_s8192_d128_causal_q1024_kv1024_c1024 | compute | valid | 1129.26 | -1.6% | +nan% |
| splash__h8_s8192_d128_causal_q128_kv128_c128 | compute | calib | 8355.48 | -0.9% | +nan% |
| splash__h8_s8192_d128_causal_q256_kv512_c256 | compute | calib | 2116.81 | +6.0% | +nan% |
| splash__h8_s8192_d128_causal_q512_kv1024_c512 | compute | valid | 1043.53 | +3.7% | +nan% |
| splash__h8_s8192_d128_causal_q512_kv512_c512 | compute | calib | 1144.87 | +2.9% | +nan% |
| splash__s2048_causal_q1024_kv1024_c512 | compute | calib | 91.41 | +2.3% | +nan% |
| splash__s2048_causal_q128_kv128_c128 | compute | calib | 547.10 | -0.6% | +nan% |
| splash__s2048_causal_q256_kv256_c256 | compute | valid | 181.65 | +4.8% | +nan% |
| splash__s2048_causal_q512_kv1024_c512 | compute | calib | 92.08 | +4.0% | +nan% |
| splash__s2048_causal_q512_kv512_c512 | compute | calib | 89.58 | +3.2% | +nan% |
| splash__s2048_full_q1024_kv1024_c512 | compute | calib | 123.02 | +3.7% | +nan% |
| splash__s2048_full_q128_kv128_c128 | compute | valid | 866.72 | +1.9% | +nan% |
| splash__s2048_full_q256_kv256_c256 | compute | valid | 279.97 | +7.5% | +nan% |
| splash__s2048_full_q512_kv1024_c512 | compute | valid | 119.40 | +4.4% | +nan% |
| splash__s2048_full_q512_kv512_c512 | compute | calib | 130.28 | +4.4% | +nan% |
| splash__s4096_causal_q1024_kv1024_c512 | compute | valid | 292.20 | +2.3% | +nan% |
| splash__s4096_causal_q128_kv128_c128 | compute | valid | 2121.80 | -0.8% | +nan% |
| splash__s4096_causal_q256_kv256_c256 | compute | valid | 676.43 | +4.5% | +nan% |
| splash__s4096_causal_q512_kv1024_c512 | compute | valid | 296.38 | +3.8% | +nan% |
| splash__s4096_causal_q512_kv512_c512 | compute | calib | 309.93 | +3.0% | +nan% |
| splash__s4096_local_q1024_kv1024_c512 | compute | valid | 204.62 | +3.4% | +nan% |
| splash__s4096_local_q128_kv128_c128 | compute | calib | 538.09 | +1.6% | +nan% |
| splash__s4096_local_q256_kv256_c256 | compute | valid | 214.81 | +6.5% | +nan% |
| splash__s4096_local_q512_kv1024_c512 | compute | valid | 169.45 | +3.8% | +nan% |
| splash__s4096_local_q512_kv512_c512 | compute | valid | 134.03 | +3.7% | +nan% |
