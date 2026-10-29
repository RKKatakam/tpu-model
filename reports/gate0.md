# Gate 0: environment, capture record, noise

## What ran

| session | kind | batch | result | balance before -> after (CU) |
|---|---|---|---|---|
| cpu-aot-1 | Colab CPU | phase0_aot (probe + 4 AOT compiles) | 4/4 ok | 218.03 -> 218.03 |
| cpu-aot-2 | Colab CPU | phase0_aot (12 AOT compiles, 3 flag sets) | 12/12 ok | 218.03 -> 218.03 |
| v5e-a | Colab v5e1 | phase0_noise (4 kernels x 3 processes x 200 reps) | 12/12 ok | 218.03 -> 217.68 |
| v5e-b | Colab v5e1 | phase0_noise | 12/12 ok | 217.68 -> 217.32 |
| v5e-c | Colab v5e1 | phase0_noise | connection lost during unpack; 0 records | 217.32 -> 216.94 |
| v5e-c2 | Colab v5e1 | phase0_noise (rerun of c) | 12/12 ok | 216.69 -> 216.48 |

v5e-c hung for 13 minutes in `colab exec` after a dropped connection, so 0.38 CU bought
nothing. `scripts/colab_batch.sh` now puts a wall-clock limit and retries on every remote
step. The 216.94 -> 216.69 gap between v5e-c and v5e-c2 is the Phase 1 CPU AOT session
that ran concurrently.

Versions in every record: jax 0.12.0.1, jaxlib 0.12.0.1, libtpu 0.0.50.1, xprof 2.24.1,
Python 3.12.3. Device: `TPU v5 lite`, `TPU_ACCELERATOR_TYPE=v5e-1`, one core.

Commands: `scripts/colab_batch.sh batches/<batch>.json <session> [--tpu v5e1]`, then
`scripts/noise_report.py data/raw/phase0_noise data/raw/phase0_aot/cpu-aot-2`.

## One complete record

`reports/gate0_example_record.json` is `data/raw/phase0_noise/v5e-a/add__p0/record.json`
with sample arrays and file lists truncated. Fields: config, kernel class, params, arg
shapes, versions, LIBTPU_INIT_ARGS and XLA_FLAGS, machine (hostname, session, timestamp,
CPU, libtpu build, devices, TPU_* env), XLA cost analysis, dump file list with sha256,
device-time samples per profiler line with stats, and the profile's plane/line/event
summary.

## Offline bundles equal on-device bundles

Final bundles from AOT compilation on a Colab CPU host (`get_topology_desc("v5e:1x1")`)
are byte-identical to the bundles compiled on the v5e, for all four kernels, across all
nine processes and three sessions. The dump and annotation flags change only comments
(`HARDWARE_FACTS.md` F2, F3).

## Noise

Medians of 200 profiled reps per process; spread is (max - min) / median.

| kernel | class | median | within-run CV | process-to-process | session-to-session (3 hosts) |
|---|---|---|---|---|---|
| add 4096x4096 f32 | streaming | 292.20 µs | 0.06-0.07% | 0.01% | 0.01% |
| col_sum 8192x4096 f32 | streaming | 178.22 µs | 0.05% | 0.01% | 0.00% |
| matmul 2048^3 bf16, 512 tiles | compute | 130.16 µs | 0.40-0.44% | 0.07% | 0.02% |
| add_one 8x128 f32 | trivial | 0.558 µs | 10-12% (tail) | 0.18% | 0.00% |

Hosts: bee1bfd566f7, e4af62c94e29, 13ac7b07e859. No drift between the first and last 50
reps of any run. `add_one` has a tail: 10 to 16% of reps run up to 1.43x the median, spread
through the run, while its MAD stays at 0.18%. Medians are the ground truth statistic.

Noise floor for claims: an improvement must exceed about 0.1% on large kernels and 0.2%
on sub-microsecond kernels. Full per-process table: `reports/gate0_noise.md`.

## Confirmed vs hypothesized

Confirmed (`HARDWARE_FACTS.md`): F1 the Colab v5e1 is one TensorCore of a TPU v5 lite;
F2 AOT bundles equal device bundles; F3 dump flags do not change instructions; F4 device
time is the `XLA Ops` event on `/device:TPU:0`; F5 XLA's Mosaic estimate comes only from
`--xla_jf_dump_static_llo_profile_to`; F6 the noise above.

Still hypotheses: clock, MXU size, slot capacities, DMA behaviour, the cause of the
`add_one` tail, whether neighbouring chips on the host interfere.

## Problems and surprises

- libtpu has no macOS wheel, so "offline" AOT runs on a Colab CPU runtime (no TPU).
- XLA's `estimated_cycles` for a Pallas kernel is not in the default dump; it needs
  `--xla_jf_dump_static_llo_profile_to`.
- One session lost its connection and idled; fixed in the driver.

## Next

Phase 1: clock fit from straight-line kernels, DMA size sweep, matmul tile configs.
