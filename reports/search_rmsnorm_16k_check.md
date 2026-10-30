| candidate | predicted us | measured us | error |
|---|---|---|---|
| rmsnorm_16k__block_rows64 | 794.75 | 813.31 | -2.3% |
| rmsnorm_16k__block_rows128 | 812.41 | 815.16 | -0.3% |
| rmsnorm_16k__block_rows32 | 848.33 | 810.35 | +4.7% |
| rmsnorm_16k__block_rows16 | 912.09 | 859.07 | +6.2% |
| rmsnorm_16k__block_rows8 | 1154.72 | 1195.59 | -3.4% |

model's pick rmsnorm_16k__block_rows64: 813.31 us; best measured rmsnorm_16k__block_rows32: 810.35 us; regret +0.4%; Spearman 0.70; median |error| 3.4%
