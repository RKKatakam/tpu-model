<img src="assets/icon.svg" width="64" height="64" alt="">

# tpu-model

A cycle-level performance model for Pallas kernels on Google TPU v5e. It replays the compiler's final
VLIW bundles on a calibrated model of one TensorCore and its DMA engine, and predicts the kernel's
device time. It is meant for autotuning: rank tile sizes and kernel variants without running them.

Every model parameter comes from a microbenchmark run on a real v5e (Colab `v5e1`) and verified
against its bundle dump. Accuracy is checked on kernels the fitting never sees, against the median
device time from the profiler, with XLA's own static estimate as the baseline.

## Status

Device time error, `predicted / measured - 1`:

| Test set | Kernels | Median abs error | Within 2% | Within 5% | Worst |
|---|---|---|---|---|---|
| Validation split (held out from fitting) | 71 | 1.1% | 49 | 67 | -10.2% (matmul) |
| Calibration split | 82 | 1.0% | 62 | 77 | -7.3% (matmul) |
| External: vLLM `tpu-inference` kernels (never fitted) | 26 | 1.0% | 18 | 22 | -8.1% (int8 matmul) |

Against XLA's static estimate on the validation split: compute kernels median 1.1% vs 9.7%;
streaming kernels 0.9% vs 31.4%. Within autotune sweeps (several tilings of one kernel), the model
picks the fastest tiling in 24 of 30 sweeps; XLA's estimate, where it accepts the kernel, picks it
in 8 of 13.

The goal is every kernel within 2%. Known gaps:

- Tiled matmuls with large tiles and strided input streams are 5-10% fast.
- Rope with small blocks is +7% to +10%. Its compute and its DMA pattern each fit on their own;
  the error comes from how they interleave.
- tgmm with 128 and 512 tiles is about -6%.
- Paged and ragged decode attention with few pages per block is about +5%. Traces show the DMA
  waits match the hardware, so the error is in core time.
- External int8 quantized matmul (w8a8) is off by up to 8%.

Scope: one v5e TensorCore, single chip, Pallas kernels. There is no multi-chip communication,
SparseCore or whole-program XLA graph modeling.

## How it works

1. **Capture** (`capture/`, `scripts/colab_batch.sh`): each kernel config runs in a fresh process on
   a Colab TPU VM with pinned versions (`VERSIONS.lock`). It records the profiler's per-rep device
   time, the HLO dumps and the final bundle dumps (`--xla_jf_dump_to`).
2. **Parse** (`parser/`): final bundles become regions, ops, operands and branch targets.
3. **Replay** (`model/machine.py`): a scalar interpreter follows the real control flow. Integer kernel
   operands (page tables, mask tables, group sizes) are rebuilt on the host and placed in SMEM, so
   data-dependent loops and DMAs take the paths they took on the hardware.
4. **Core timing** (`model/core.py`): per-slot issue queues with decoupled vector issue, a scoreboard
   with measured result latencies, MXU/XLU/EUP result FIFOs (push now, pop later), transposer
   streams, scalar-load latency and VMEM store-to-load waits.
5. **DMA timing** (`model/dma.py`): one FIFO per direction, read/write interleaving, start-up curves
   by size, a per-piece cost for strided copies, and semaphore waits.
6. **Instruction overlays** (`model/overlay.py`): programs larger than instruction memory pay chunk
   switches and loop-wrap reloads.

`HARDWARE_FACTS.md` lists every confirmed hardware fact with the experiment and data behind it, and
keeps hypotheses separate. The model may only depend on confirmed facts.

## Methodology

- **Splits:** records are split into calibration and validation by a hash of the record id, fixed
  in advance. Microbenchmarks and the calibration split are used for fitting; validation and
  external records never are.
- **External set:** kernels vendored from vllm-project/tpu-inference (`corpus/external`, Apache 2.0,
  pinned commit in `SOURCE`), using that repository's own test shapes.
- **Ground truth:** median device time from the profiler. In-kernel tracing
  (`TRACE_COMPUTE_AND_SYNC` sync-flag events) is used only to diagnose DMA waits and instruction
  overlays.
- **Noise** was measured before any claim (`scripts/noise_report.py`). Improvements smaller than
  noise are not reported.

## Layout

| Path | Contents |
|---|---|
| `bench/` | Microbenchmark kernels (`micro.py`) and bundle-dump checks that each one isolates what it claims (`checks.py`) |
| `corpus/` | Kernel builders: splash, flash, paged and ragged attention, matmul, tgmm/gmm, layernorm, softmax, rmsnorm, rope, PRNG, and the external vLLM kernels |
| `capture/` | On-TPU capture: timing, profiles, dumps, deviceless AOT compiles |
| `parser/` | Bundle and dump parsing |
| `model/` | Replay, core, DMA and overlay models; fitted parameters in `model/params/` |
| `scripts/` | Batch definitions, fitting, validation, external report, sync-trace diff, search |
| `reports/` | Latest validation and external reports |
| `batches/` | Generated batch configs |
| `data/` | Raw records and dumps (not in git) |

## Usage

Local environment (macOS or Linux, CPU only):

```sh
uv sync
uv run pytest -q
```

Run a batch on Colab (needs the Colab CLI; TPU access only via Colab):

```sh
python scripts/make_batches.py                       # writes batches/*.json
scripts/colab_batch.sh batches/phase4a.json v5e-4a --tpu v5e1
```

Score the model:

```sh
python scripts/validate.py reports/validation.md data/raw/corpus*        # both splits, plus XLA
python scripts/eval_calib.py model/params/core.json model/params/dma.json data/raw/corpus*
python scripts/report_external.py reports/external_vllm.md data/raw/external_vllm_*
python scripts/sync_diff.py data/raw/diag_sync RECORD_ID                 # model vs hardware DMA waits
```

## Tiling search

`scripts/search.py` has three steps:

1. `search.py plan SPACE` compiles every candidate deviceless (topology `v5e:1x1`) on a Colab CPU
   runtime. No TPU is needed: deviceless compiles give bundles identical to on-TPU compiles (checked
   on 8 kernels).
2. `search.py rank SPACE` ranks the candidates with the model.
3. `search.py check SPACE` times the top few and some controls on a TPU, and reports regret and rank
   correlation.

Seven spaces have been checked on hardware (`reports/search_summary.md`): dense matmul, splash
attention, int8 quantized matmul (w8a8 and w8a16), tgmm, rope and RMSNorm. In each, the model's pick
was the measured best tiling or within 0.7% of it.

## Versions

jax / jaxlib 0.12.0.1, libtpu 0.0.50.1, xprof 2.24.1, Python 3.12 (`VERSIONS.lock`,
`requirements.lock`).
