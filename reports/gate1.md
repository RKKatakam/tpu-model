# Gate 1: feasibility

## What ran

| session | batch | result | balance before -> after (CU) |
|---|---|---|---|
| cpu-p1-1 | phase1_aot (51 AOT compiles for bundle checks) | 51/51 ok | 216.94 -> 216.69 (concurrent with v5e-c2) |
| v5e-p1-1 | phase1_device (51 configs, 100 reps each) | 51/51 ok | 216.48 -> 215.75 |

Versions as in Gate 0. Bundle checks (`python -m bench.checks data/raw/phase1_device`): 47/47
microbenchmarks pass after allowing the end-of-kernel semaphore check (`vsyncmov` + `vpop.sfrf`)
and the compiler fully unrolling `valu_loop` trip=1. Report: `reports/phase1.md`, plots
`reports/phase1_clock.png`, `reports/phase1_dma.png`.

## Numbers

(a) Clock: **1.4999 GHz, 0 ns fixed overhead**, residual under 0.7 ns from 1 to 8201 bundles.
Loops cost exactly their issued bundles. A dependent vadd.f32 chain runs at **2 cycles per
bundle**: the hardware interlocks on a 2-cycle result latency that the bundle dump does not show.

(b) DMA, contiguous HBM->VMEM, one at a time: smooth, monotone, repeatable. 301 ns at 512 B,
334 ns at 4 KiB, 1114 ns at 512 KiB, 5972 ns at 4 MiB (702 GB/s).

(c) Matmul 2048^3 bf16, prediction from (a) + (b) with a minimal replay (scalar interpreter,
one cycle per bundle, DMAs on a single FIFO HBM pipe using the (b) table):

| tiles | measured µs | ours | ours err | XLA est | XLA err |
|---|---|---|---|---|---|
| 1024,1024,512 | 103.97 | 84.58 | -18.7% | 100.17 | -3.7% |
| 512,512,512 | 130.21 | 120.02 | -7.8% | 106.08 | -18.5% |
| 256,512,1024 | 157.01 | 166.69 | +6.2% | 124.27 | -20.9% |
| 256,256,256 | 311.67 | 298.84 | -4.1% | 332.53 | +6.7% |

Ranking: measured, ours, and XLA all order the four configs the same way.

## Gate 1: FAIL (worst -18.7%, needed about 10%)

What went wrong, with evidence:

1. **Interlock stalls are missing.** (a) shows the hardware stalls on operand latency and the
   dump does not show it. In the 1024-tile config the replay issues 112k bundles plus 15k DMA
   stall cycles = 127k cycles, but the device took 156k. Each MXU receives a 16-row `vmatmul`
   every 3 bundles, faster than 128-wide MXU throughput, so MXU occupancy and `vpop` result
   latency must stall issue.
2. **DMA concurrency is wrong.** The config with the most DMA traffic (256,512,1024) is
   over-predicted (+6.2%) because overlapping transfers are serialized on one pipe; (b) only
   measured one transfer at a time.

Neither fix is kernel-specific; both are Phase 3 calibration items (register scoreboard with
measured latencies, MXU occupancy/latency, DMA concurrency and VMEM->HBM curves).

## Decision

The owner asked to keep going until the model is accurate, so I continue past this gate
rather than stopping. Phase 3 microbenchmarks target exactly the two causes above. No
parameter was tuned against the matmul results.
