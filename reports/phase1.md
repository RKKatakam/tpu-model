## 1a. Clock from straight-line kernels

Fit over 7 hazard-free kernels (nv=8 independent vadd chains, no loops, no DMA): time = a + bundles / f.

- f = **1.4999 GHz**, a = **-0.03 ns** fixed overhead
- residuals: max |r| = 0.69 ns

| record | bundles | measured ns | a + bundles/f | measured cycles / bundle |
|---|---|---|---|---|
| valu_straight__nv1_k0 | 1 | 1.0 | 0.6 | 1.500 |
| valu_straight__nv1_k64 | 66 | 87.0 | 44.0 | 1.977 |
| valu_straight__nv1_k256 | 258 | 343.0 | 172.0 | 1.994 |
| valu_straight__nv1_k1024 | 1026 | 1367.0 | 684.0 | 1.998 |
| valu_straight__nv1_k4096 | 4098 | 5463.0 | 2732.2 | 2.000 |
| valu_straight__nv8_k0 | 1 | 1.0 | 0.6 | 1.500 |
| valu_straight__nv8_k32 | 73 | 48.0 | 48.6 | 0.986 |
| valu_straight__nv8_k128 | 265 | 177.0 | 176.7 | 1.002 |
| valu_straight__nv8_k512 | 1033 | 688.0 | 688.7 | 0.999 |
| valu_straight__nv8_k1024 | 2057 | 1372.0 | 1371.4 | 1.000 |
| valu_straight__nv8_k2048 | 4105 | 2737.0 | 2736.8 | 1.000 |
| valu_straight__nv8_k4096 | 8201 | 5467.5 | 5467.7 | 1.000 |

nv=1 (each vadd.f32 reads the previous bundle's result): 2.000 cycles per bundle. The dump has no padding between them, so the hardware interlocks: **vadd.f32 result latency = 2 cycles**.

## Loops: does a taken branch cost more than its bundles?

| record | bundles issued | measured ns | a + bundles/f | extra ns per iteration |
|---|---|---|---|---|
| valu_loop__k1_t1000 | 8006 | 5337.0 | 5337.7 | -0.001 |
| valu_loop__k1_t10000 | 80006 | 53337.0 | 53340.8 | -0.000 |
| valu_loop__k8_t1 | 25 | 17.0 | 16.6 | +0.358 |
| valu_loop__k8_t10 | 189 | 126.0 | 126.0 | +0.002 |
| valu_loop__k8_t100 | 1809 | 1206.0 | 1206.1 | -0.001 |
| valu_loop__k8_t1000 | 18009 | 12006.0 | 12006.8 | -0.001 |
| valu_loop__k8_t10000 | 180009 | 120006.0 | 120013.7 | -0.001 |

## 1b. Single-stream HBM->VMEM DMA vs size

Per-copy time = (t(8 copies) - t(1 copy)) / 7, minus 3 bundles of issue/wait/syncadd at f.

| bytes | t(1) ns | t(8) ns | per-copy ns | GB/s |
|---|---|---|---|---|
| 512 | 317.0 | 2441.0 | 301.4 | 1.7 |
| 1024 | 322.0 | 2487.5 | 307.4 | 3.3 |
| 2048 | 325.0 | 2568.5 | 318.5 | 6.4 |
| 4096 | 335.0 | 2690.5 | 334.5 | 12.2 |
| 8192 | 338.0 | 2821.0 | 352.7 | 23.2 |
| 16384 | 351.0 | 2958.5 | 370.5 | 44.2 |
| 32768 | 381.0 | 3219.5 | 403.5 | 81.2 |
| 65536 | 422.0 | 3564.5 | 446.9 | 146.6 |
| 131072 | 538.0 | 4528.0 | 568.0 | 230.8 |
| 262144 | 717.0 | 6060.5 | 761.4 | 344.3 |
| 524288 | 1060.0 | 8874.0 | 1114.3 | 470.5 |
| 1048576 | 1763.0 | 14604.0 | 1832.4 | 572.2 |
| 2097152 | 3181.0 | 25761.5 | 3223.8 | 650.5 |
| 4194304 | 5951.5 | 47771.0 | 5972.2 | 702.3 |

Fit on sizes >= 64 KiB: per-copy = L + bytes / B with L = **403.9 ns**, B = **750.4 GB/s** (max |residual| 44.3 ns).
Copies <= 4 KiB take 301.4 to 334.5 ns each.

## 1c. Double-buffered matmul 2048^3 bf16, prediction vs measured

Parameters come only from 1a (clock, overhead) and 1b (per-copy DMA time table). Vector/MXU ops cost one issue cycle each; interlock stalls are not modelled yet. XLA's estimate is converted with the fitted clock.

| tiles (bm,bn,bk) | measured µs | ours µs | ours err | XLA est µs | XLA err | bundles issued | DMA stall cycles |
|---|---|---|---|---|---|---|---|
| 1024,1024,512 | 103.97 | 84.58 | -18.7% | 100.17 | -3.7% | 112034 | 14822 |
| 256,256,256 | 311.67 | 298.84 | -4.1% | 332.53 | +6.7% | 227822 | 220409 |
| 256,512,1024 | 157.01 | 166.69 | +6.2% | 124.27 | -20.9% | 134158 | 115857 |
| 512,512,512 | 130.21 | 120.02 | -7.8% | 106.08 | -18.5% | 135486 | 44539 |

Gate 1 (within ~10% on every config): **FAIL** (worst -18.7%).
