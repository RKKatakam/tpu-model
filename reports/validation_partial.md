# Validation

Error = predicted / measured - 1 on the median device time. XLA = `estimated_cycles` from `--xla_jf_dump_static_llo_profile_to` at 1.5 GHz.

## Per-class error, split = valid

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 3 | 2.5% | 8.4% | 9.9% | 6.2% | 13.1% | 14.8% | 2/3 | 3/3 |
| streaming | 7 | 3.7% | 7.3% | 9.0% | 42.4% | 52.2% | 52.3% | 4/7 | 7/7 |
| latency | 0 | - | - | - | - | - | - | 0/0 | 0/0 |

## Per-class error, split = calib

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 2 | 8.5% | 8.6% | 8.7% | 10.5% | 17.3% | 19.0% | 0/2 | 2/2 |
| streaming | 9 | 10.3% | 19.1% | 20.0% | 49.0% | 73.1% | 73.2% | 1/9 | 4/9 |
| latency | 0 | - | - | - | - | - | - | 0/0 | 0/0 |

## Per-class error, supplementary kernels (flash), all splits

| class | n | ours median | ours p90 | ours worst | XLA median | XLA p90 | XLA worst | ours within 5% | ours within 10% |
|---|---|---|---|---|---|---|---|---|---|
| compute | 15 | 4.4% | 6.7% | 8.8% | 4.1% | 31.3% | 35.1% | 9/15 | 15/15 |
| streaming | 0 | - | - | - | - | - | - | 0/0 | 0/0 |
| latency | 0 | - | - | - | - | - | - | 0/0 | 0/0 |

## Ranking within autotune sweeps (all splits)

| sweep | n | Spearman ours | Spearman XLA | top-1 ours | top-1 XLA | top-3 overlap ours | regret ours | regret XLA |
|---|---|---|---|---|---|---|---|---|
| flash{"b": 1, "causal": false, "d": 128, "h": 8, "s": 2048} | 5 | 1.00 | 1.00 | yes | yes | 3/3 | 0.0% | 0.0% |
| flash{"b": 1, "causal": true, "d": 128, "h": 8, "s": 2048} | 5 | 1.00 | 0.90 | yes | no | 3/3 | 0.0% | 2.1% |
| flash{"b": 1, "causal": true, "d": 128, "h": 8, "s": 4096} | 5 | 0.90 | 0.90 | yes | yes | 3/3 | 0.0% | 0.0% |
| layernorm{"cols": 1024, "rows": 8192} | 3 | 1.00 | 1.00 | yes | yes | 3/3 | 0.0% | 0.0% |
| matmul{"k": 4096, "m": 4096, "n": 4096} | 3 | 1.00 | 1.00 | yes | yes | 3/3 | 0.0% | 0.0% |
| softmax{"cols": 1024, "rows": 8192} | 3 | 1.00 | 1.00 | yes | yes | 3/3 | 0.0% | 0.0% |

## All records

| id | class | split | measured µs | ours | XLA |
|---|---|---|---|---|---|
| add__r2048_c2048_b64x2048 | streaming | calib | 75.51 | +8.0% | -61.6% |
| add__r8192_c2048_b1024x512 | streaming | calib | 295.55 | +10.3% | -73.2% |
| add__r8192_c2048_b256x2048 | streaming | calib | 294.16 | +11.4% | -73.1% |
| col_sum__r16384_c2048_b128 | streaming | calib | 178.03 | +16.8% | -17.8% |
| col_sum__r16384_c2048_b512 | streaming | calib | 178.20 | +4.8% | -39.3% |
| col_sum__r4096_c8192_b64 | streaming | valid | 178.16 | +9.0% | -31.4% |
| flash__s2048_c0_q1024_k512 | compute | calib | 130.94 | +1.6% | +0.3% |
| flash__s2048_c0_q128_k128 | compute | valid | 976.90 | -4.0% | +35.1% |
| flash__s2048_c0_q256_k256 | compute | valid | 312.79 | -1.5% | +24.1% |
| flash__s2048_c0_q512_k1024 | compute | calib | 130.96 | +3.7% | +0.4% |
| flash__s2048_c0_q512_k512 | compute | valid | 137.53 | +1.9% | +3.0% |
| flash__s2048_c1_q1024_k512 | compute | valid | 111.47 | +7.0% | +0.4% |
| flash__s2048_c1_q128_k128 | compute | valid | 663.29 | -4.4% | +32.3% |
| flash__s2048_c1_q256_k256 | compute | calib | 220.99 | +0.9% | +27.2% |
| flash__s2048_c1_q512_k1024 | compute | valid | 114.67 | +4.4% | +2.5% |
| flash__s2048_c1_q512_k512 | compute | valid | 109.14 | +8.8% | +5.8% |
| flash__s4096_c1_q1024_k512 | compute | calib | 363.26 | +6.3% | +0.2% |
| flash__s4096_c1_q128_k128 | compute | calib | 2533.14 | -5.5% | +29.8% |
| flash__s4096_c1_q256_k256 | compute | calib | 810.66 | -1.0% | +23.8% |
| flash__s4096_c1_q512_k1024 | compute | valid | 372.52 | +5.1% | +2.1% |
| flash__s4096_c1_q512_k512 | compute | calib | 370.19 | +6.1% | +4.1% |
| layernorm__r16384_c4096_b64 | streaming | calib | 816.85 | +6.3% | -49.0% |
| layernorm__r16384_c4096_b8 | streaming | calib | 1285.88 | -20.0% | -0.6% |
| layernorm__r8192_c1024_b256 | streaming | valid | 102.47 | +6.2% | -52.1% |
| layernorm__r8192_c1024_b64 | streaming | valid | 110.46 | +0.3% | -18.5% |
| layernorm__r8192_c1024_b8 | streaming | valid | 366.40 | -1.4% | +44.7% |
| matmul__n1024_256_256_256 | compute | calib | 41.79 | -8.3% | +2.1% |
| matmul__n1024_512_512_1024 | compute | calib | 18.21 | +8.7% | -19.0% |
| matmul__n4096_1024_1024_512 | compute | valid | 755.73 | +1.1% | -1.1% |
| matmul__n4096_256_1024_512 | compute | valid | 1037.78 | +9.9% | -6.2% |
| matmul__n4096_512_512_512 | compute | valid | 956.10 | -2.5% | -14.8% |
| softmax__r16384_c4096_b64 | streaming | calib | 818.47 | +5.9% | -53.9% |
| softmax__r16384_c4096_b8 | streaming | calib | 1228.53 | -18.8% | +1.0% |
| softmax__r8192_c1024_b256 | streaming | valid | 102.53 | +6.1% | -52.3% |
| softmax__r8192_c1024_b64 | streaming | valid | 109.64 | -0.1% | -18.7% |
| softmax__r8192_c1024_b8 | streaming | valid | 367.99 | -3.7% | +42.4% |
