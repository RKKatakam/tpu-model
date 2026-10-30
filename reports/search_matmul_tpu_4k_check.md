| candidate | predicted us | measured us | error |
|---|---|---|---|
| matmul_tpu_4k__bm512_bn1024_bk1024 | 748.57 | 754.90 | -0.8% |
| matmul_tpu_4k__bm1024_bn512_bk1024 | 751.71 | 754.00 | -0.3% |
| matmul_tpu_4k__bm1024_bn1024_bk512 | 752.24 | 749.29 | +0.4% |
| matmul_tpu_4k__bm1024_bn1024_bk256 | 787.93 | 779.21 | +1.1% |
| matmul_tpu_4k__bm512_bn1024_bk512 | 792.82 | 799.20 | -0.8% |
| matmul_tpu_4k__bm1024_bn512_bk512 | 794.20 | 793.49 | +0.1% |
| matmul_tpu_4k__bm1024_bn1024_bk128 | 839.66 | 830.09 | +1.2% |
| matmul_tpu_4k__bm512_bn512_bk256 | 1083.57 | 1088.85 | -0.5% |
| matmul_tpu_4k__bm256_bn256_bk1024 | 1575.96 | 1549.11 | +1.7% |
| matmul_tpu_4k__bm128_bn512_bk512 | 1988.38 | 1999.76 | -0.6% |
| matmul_tpu_4k__bm128_bn256_bk512 | 2712.06 | 2808.34 | -3.4% |
| matmul_tpu_4k__bm128_bn128_bk256 | 6315.72 | 6702.17 | -5.8% |

model's pick matmul_tpu_4k__bm512_bn1024_bk1024: 754.90 us; best measured matmul_tpu_4k__bm1024_bn1024_bk512: 749.29 us; regret +0.7%; Spearman 0.97; median |error| 0.8%
