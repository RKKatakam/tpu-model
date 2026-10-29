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

### F7. Clock is 1.5 GHz and hazard-free code issues one bundle per cycle with no launch overhead

- Experiment: `bench/micro.py:valu_straight` nv=8, k in {0..4096} (bundle check `bench/checks.py`:
  exactly k*nv vadd.f32, no DMA, no branch, no spill). `batches/phase1_device.json`, session v5e-p1-1.
  Fit: `scripts/phase1_report.py`. Data: `data/raw/phase1_device/v5e-p1-1/valu_straight__*`.
- Result: device time = bundles / 1.4999 GHz - 0.03 ns, max residual 0.69 ns over 1 to 8201 bundles.
  The `XLA Ops` event of a 1-bundle kernel lasts 1.0 ns.
- Confidence: high. Confirms H1.

### F8. A taken backward branch costs only its bundles (4 delay slots, no extra penalty)

- Experiment: `bench/micro.py:valu_loop` k in {1, 8}, trip 10 to 10000 (bundle check: k*nv vadd, one
  loop branch). Same session and script.
- Result: valu_loop k8 t10000 issues 180009 bundles (replayed by `model/machine.py`) and takes
  120006 ns = 180009 cycles at 1.5 GHz; k1 t10000: 80006 bundles, 53337 ns = 80006 cycles.
- Confidence: high.

### F9. vadd.f32 has a 2-cycle result latency enforced by a hardware interlock

- Experiment: `valu_straight` nv=1 (each vadd reads the previous bundle's result; the dump has one
  vadd per bundle and no padding), k in {64, 256, 1024, 4096}.
- Result: 2.000 cycles per bundle (k=4096: 4098 bundles, 5463 ns). The static bundle count
  undercounts such code; the model needs a register scoreboard.
- Confidence: high for vadd.f32. Other ops: Phase 3.

### F10. Single-stream HBM->VMEM DMA time vs size (start, then wait, one at a time)

- Experiment: `bench/micro.py:dma_seq` rows in {1..8192} x 128 f32 (512 B to 4 MiB), 1 and 8 copies
  (bundle check: exactly `reps` dma.hbm_to_vmem of the right granule count, one wait each, no
  vector work besides the final semaphore check). Per-copy = (t8 - t1) / 7.
- Result: 301 ns at 512 B, 334 ns at 4 KiB, 447 ns at 64 KiB, 1114 ns at 512 KiB, 5972 ns at 4 MiB
  (702 GB/s). Smooth and monotone; table in `reports/phase1.md`, stored in `model/params/phase1.json`.
- Confidence: high for contiguous, single stream. Concurrency, strides, VMEM->HBM: Phase 3.

### F11. Vector result latencies: vmul.f32 and vmax.f32 = 2 cycles; EUP exp = 6 cycles; XLU = 106-107 cycles

- Experiment: `bench/micro.py:vop_chain` (one dependent op per bundle, nv=1; bundle check
  `bench/checks.py`), `batches/phase3a.json`, session v5e-p3a. Fit: `scripts/fit_core.py`
  (`model/params/core.json` `fit_report`).
- Result: vmul/vmax 2.0 cycles per dependent op like vadd (F9). `vexp2` + `vpop.eup` chain 6.0
  cycles. Lane rotate (`vrot.lane` + `vpop.permute`) 106.0 and lane reduce (`vadd.xlane` +
  `vpop.xlane`) 107.0 cycles from issue to popped result.
- Confidence: high for these ops. Cast and sublane rotate chains were folded by the compiler and
  measured nothing.

### F12. MXU (bf16): 16 cycles of occupancy per vmatmul, two result vregs at ~123 and ~133 cycles

- Experiment: `bench/micro.py:mxu_dot` (acc += x @ w in VMEM, no DMA) over 10 shapes, reps 4 and
  16, `batches/phase3a.json`. Fit on reps 4, checked on reps 16 (`scripts/fit_core.py`).
- Result: 1024 -> 4096 rows adds 16.0 cycles per extra vmatmul per MXU (one row per cycle, 16-row
  packed bf16 vreg). Fitted: first result 123.0 cycles after issue, second 10.0 later, weight push
  4.0 cycles per `vmatpush`, 0.3 cycles from last push to a `vlgmr` load. Check (reps 16): all
  within 1.7%.
- Confidence: high for these shapes. Caveat: the 512-wide shapes spill vregs (bench.checks), so
  those records also exercise spill loads/stores.

### F13. MXU (f32 LHS): 8 cycles per vmatmul, one result at ~125 cycles; f32 weight pushes cost 2 cycles

- Experiment: `mxu_dot` with lhs f32 and rhs bf16 or f32, 7 shapes, reps 4 and 16,
  `batches/phase3d.json`, session v5e-p3d (bundle check: only `vmatmul.f32.*`, no spill).
- Result: 8.0 cycles per extra vmatmul per MXU (8-row f32 vreg, same row rate as bf16); 32
  `vmatmul.f32` pop 32 results per MXU. f32 weights take twice the `vmatpush` ops of bf16 yet the
  same time; fitted push cost 2.00 cycles, result latency 124.9. All 28 configs within 2.0%.
- Confidence: high for these shapes.

### F14. HBM sustained bandwidth under the Pallas pipeline: 751 GB/s read, 655 GB/s write, mixed slower

- Experiment: `bench/micro.py:dma_stream` (pipelined grid over a 16384 x 1024 f32 array, reads in,
  writes out, vld/vst/vadd only; bundle check), blocks 32/128/512 rows, `batches/phase3c.json`,
  session v5e-p3c2.
- Result (128- and 512-row blocks): reads only 751 GB/s, writes only 655 GB/s. Mixed traffic
  fits time = R/751 + W/655 + 0.20 ps x min(R, W) bytes within 0.3% (1:1 654, 2:1 673-684,
  1:2 641-655 GB/s); the overlap cost per byte does not change from 128- to 512-row blocks.
  32-row blocks (128 KiB) run below bandwidth (reads 448 GB/s).
- Confidence: high for contiguous blocks. Random addresses: phase 3e.

### F15. Concurrent and read-plus-write DMA, strides, single-stream writes

- Experiment: `dma_conc` (1-8 concurrent reads), `dma_inout` (a read and a write in flight),
  `dma_strided` (512 B rows at strides of 1-32 KiB), `dma_out` (single-stream writes),
  `batches/phase3b.json`, session v5e-p3b; bundle checks pass 62/62.
- Result: write per-copy curve has lower start-up than reads and ~657 GB/s at large sizes; concurrent
  reads share ~740 GB/s; a read and a write started together cost ~100-150 ns more than the model of
  two independent streams; strided reads cost at most 15% more than contiguous.
- Confidence: medium (the start-up terms are fitted, `scripts/fit_dma.py`).

### F16. Scalar prefetch operands reach SMEM by DMA at kernel entry; operand registers address 512 B units

- Experiment: bundle dumps of paged, gmm and splash (`data/raw/corpus1_rerun`), replayed by
  `model/machine.py` with operand values rebuilt by `model/operands.py`.
- Result: integer tables are copied by `dma.hbm_to_smem` or `dma.vmem_to_smem` in 32-byte granules
  (arrays padded to 512 B: s32[8], s32[16] and s32[32] all move 16 granules) before the first
  `dma.done.wait`; `sld` reads
  them word-addressed. Scalar operands arrive by value in their register. Operand registers hold
  HBM addresses in 512-byte units (an 8 MiB operand's bounds check is base + 16384); DMA pointers
  are that << 4 (32-byte granules) and are bounds-checked as signed values.
- Confidence: high for these kernels (replays issue the measured kernels' DMA counts and no
  `shalt.err`).

## Hypotheses (not yet tested; the model must not depend on these)

- H2. MXU is 128x128.
- H3. Compiler per-bundle slot capacities, from `*-final_hlo-static-per-bundle-utilization.txt`:
  MXU 4, XLU 3, VALU 4, VPOP 2, EUP 1, VLOAD 3, VLOAD:FILL 3, VSTORE 1, VSTORE:SPILL 1, SALU 2.
  This is the compiler's model, not a timing measurement.
- H4. The Colab host has four v5e chips and the VM sees one (`TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`,
  `TPU_HOST_BOUNDS=1,1,1`). Unknown whether neighbours share anything that affects timing.
- H5. Number of DMA engines and transfers in flight; whether random addresses slow small DMAs
  (paged attention runs ~45% slower than the replay at 4 KiB pages; `dma_gather`, phase 3e).
- H8. XLU occupancy per op (fixed at 1 cycle: the nv=8 chains are latency-bound and cannot
  identify it).
- H9. Transposed weight pushes (`vmatpush.bf16.xpose`, used by splash for K) cost the same as plain
  bf16 pushes. Unmeasured; the model assumes it.
- H10. DMA queueing structure: one FIFO per direction, heads interleaving (`model/dma.py:DirFifoHbm`).
  Chosen because it fits F14/F15 better than processor sharing or one FIFO; not observed directly.
- H6. Meaning of each bundle mnemonic and slot.
- H7. `add_one` slow outliers (12 to 16% of reps, up to 1.43x the median, spread through the
  run) come from something outside the kernel (refresh, host interference). Untested.
