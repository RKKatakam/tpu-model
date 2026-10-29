# Hardware and toolchain facts

Rule: the model may depend only on entries under "Confirmed". Each entry names the
experiment (script and data path), the result, and a confidence level. Data paths are
under `data/` (gitignored, not published).

Versions for every entry unless noted: jax 0.12.0.1, jaxlib 0.12.0.1, libtpu 0.0.50.1,
xprof 2.24.1, Python 3.12.3 (see `VERSIONS.lock`).

## Confirmed

### F1. Colab `v5e1` exposes one TensorCore of a "TPU v5 lite" chip

- Experiment: `capture/capture_one.py` via `scripts/colab_batch.sh batches/phase0_noise.json <session> --tpu v5e1`.
  Data: `data/raw/phase0_noise/v5e-*/*/record.json` (`machine` field).
- Result: `jax.devices()` = 1 device, `device_kind="TPU v5 lite"`, `num_cores=1`,
  `TPU_ACCELERATOR_TYPE=v5e-1`, platform `TFRT TPU v5 lite ... cl/1004524046`.
- Confidence: high for "it is a v5e chip with one TensorCore". Not tested: whether it
  behaves identically to a Cloud TPU v5e.

### F2. Final bundles from AOT compile on a CPU host match the on-device compile byte for byte

- Experiment: `batches/phase0_aot.json` on Colab CPU sessions `cpu-aot-1`, `cpu-aot-2`
  (`get_topology_desc("v5e:1x1", "tpu")`), against `batches/phase0_noise.json` on v5e.
  Compared with `scripts/noise_report.py` and `scripts/compare_bundles.py`.
- Result: for `add_one`, `add`, `col_sum`, `matmul`, the `*-final_bundles.txt` sha256 is the
  same on the CPU host and on every v5e process and session. Two CPU sessions also match
  each other byte for byte.
- Confidence: high for these four kernels. Caveat: byte equality requires the same source
  path (`/content/tpum/...`), because bundle comments embed source locations.

### F3. The dump and annotation flags do not change the generated instructions

- Experiment: same kernels compiled AOT with the full flag set (`aot__full`), full plus
  `--xla_jf_dump_static_llo_profile_to` and `--xla_jf_dump_llo_critical_paths_to`
  (`aot__estimate`), and only `--xla_jf_dump_to --xla_jf_dump_llo_text` (`aot__minimal`).
  Data: `data/raw/phase0_aot/cpu-aot-2/`. Script: `scripts/compare_bundles.py`.
- Result: full vs estimate are byte-identical. Full vs minimal differ only inside `/* */`
  comments; comment-stripped bundle lines are identical (13, 587, 429, 2424 bundles).
- Confidence: high for these four kernels.

### F4. Device time of a Pallas kernel is the `XLA Ops` event on plane `/device:TPU:0`

- Experiment: `capture/xplane.py` `summarize()` output stored in each record
  (`timing.profile_summary`). Data: `data/raw/phase0_noise/v5e-a/*/record.json`.
- Result: lines on `/device:TPU:0` are `Async XLA Ops`, `Pallas Primitives`, `TC Overlay`,
  `XLA Modules`, `XLA Ops`, `_counters_`. The kernel appears once per rep on `XLA Ops`,
  named `%k_<name>.1 = ... custom-call(...)`. The enclosing `XLA Modules` event is 16 ns
  longer for `add`. `ProfileData` field names `plane.lines`, `line.events`, `ev.name`,
  `ev.start_ns`, `ev.duration_ns` exist in jax 0.12.0.1 (durations are float ns).
- Confidence: high.

### F5. XLA's static estimate for a Mosaic kernel comes from `--xla_jf_dump_static_llo_profile_to`

- Experiment: `aot__estimate` configs above; parsed by `parser/dumps.py:static_estimated_cycles`.
- Result: `<kernel>.1_TPU v5 lite_static_llo_profile.textproto` holds `hlo_op_estimate {
  estimated_cycles: N }`. The `estimated_cycles` field in `after_codegen.txt` backend configs
  exists only for XLA's own fusions, never for the `tpu_custom_call`.
- Confidence: high (toolchain fact).

### F6. Measurement noise (see `reports/gate0.md` for the full table)

- Experiment: `batches/phase0_noise.json`: 4 kernels x 3 fresh processes x 200 profiled reps
  (20 warmup), on separate Colab v5e sessions. Script: `scripts/noise_report.py`.
- Confidence: high for the sessions measured; see the gate report for numbers.

## Hypotheses (not yet tested; the model must not depend on these)

- H1. Clock about 1.5 GHz (197e12 / (4 x 128 x 128 x 2)). Phase 1a measures it.
- H2. MXU is 128x128.
- H3. Compiler per-bundle slot capacities, from `*-final_hlo-static-per-bundle-utilization.txt`:
  MXU 4, XLU 3, VALU 4, VPOP 2, EUP 1, VLOAD 3, VLOAD:FILL 3, VSTORE 1, VSTORE:SPILL 1, SALU 2.
  This is the compiler's model, not a timing measurement.
- H4. The Colab host has four v5e chips and the VM sees one (`TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`,
  `TPU_HOST_BOUNDS=1,1,1`). Unknown whether neighbours share anything that affects timing.
- H5. Number of DMA engines, transfers in flight, stride and size penalties, HBM contention.
- H6. Meaning of each bundle mnemonic and slot.
- H7. `add_one` slow outliers (12 to 16% of reps, up to 1.43x the median, spread through the
  run) come from something outside the kernel (refresh, host interference). Untested.
