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

### F17. A dependent scalar-load chain takes one cycle per iteration more than its bundles

- Experiment: `bench/micro.py:sld_chain` (j = table[j] over a scalar-prefetched cyclic table; 8 bundles
  per iteration, the loaded value is moved into the next load's address register 6 bundles after the
  load), n 256 and 4096, `batches/phase3f.json`, session v5e-p3f.
- Result: 9.0 cycles per iteration against 8 issued bundles (both n).
- Confidence: high for the measurement. The mechanism is open (H11), so the model does not use it.

### F18. Transposed bf16 weight pushes (`vmatpush.bf16.xpose`) cost ~7.9 cycles, twice a plain push

- Experiment: `mxu_dot` with weights stored (n, k) and contracted on their last dim, 3 shapes, reps 4
  (fit) and 16 (check), `batches/phase3f.json`.
- Result: push cost 7.91 cycles; fit and check within 0.1%. Splash pushes K this way.
- Confidence: high for these shapes. Refutes the former hypothesis H9.

### F19. Starting a DMA and waiting on one that has already finished cost the core nothing extra

- Experiment: `bench/micro.py:dma_overhead` (start an (8,128) copy, 512 dependent vadds, wait) vs the
  same loop without the copy, n 64 and 256, `batches/phase3f.json`.
- Result: 176650 vs 176646 ns (n=256), 44170 vs 44166 ns (n=64): the start, wait and semaphore
  update add only their bundles. The replay matches within 1%.
- Confidence: high.

### F20. A VMEM load of an address stored within 10 cycles waits for the store

- Experiment: `bench/micro.py:vmem_access` plain/masked forms (phase3k, a vst then a vld of the same
  VMEM address on the next pass) and `vmem_chain` (phase3i, the same through run-time addresses).
- Result: one store->load latency of 10 cycles fits all three (plain +0.6..3.3%, masked +0.1%,
  vmem_chain +-1.2%); without it the replay is 18-62% fast. Compiled kernels rarely hit it.
- Confidence: high.

### F21. The XLU transposer streams: one input vreg per ~8 cycles (f32), result j ~160 cycles after input j

- Experiment: `xpose_chain` (phase3j, dependent 128x128 transposes, f32 and bf16) and `tgmm_tile`
  (phase3n, megablox tgmm's inner step, transposes beside weight pushes and matmuls, masked and not).
- Result: per form, acceptance interval and per-result latency: f32 (b32) 8.2 / 160 cycles; packed
  bf16 (c.b16) 16.6 / 169 (twice the rows per vreg, same row rate); bf16 binary 8.5 / 164. All 13
  transpose benchmarks within 2.5%. A model where results appear only after the last input fits
  either context but not both (chains +7..10% with tgmm steps still -11..-18%).
- Confidence: high for these forms.

### F22. Masked weight pushes (`vmatpush.*.msk`) cost the same as plain pushes

- Experiment: `tgmm_tile` masked vs unmasked (phase3n; bundle check confirms vmatpush.bf16.msk).
- Result: 14715 vs 14492 cycles (128 tiles, 32 reps); 19132 vs 18909 (256 tiles).
- Confidence: high for bf16 masked pushes.

### F23. Vector ALU result latencies: most ops 2 cycles, select 1
- Experiment: `vop_chain2` with nv=1 (latency) and nv=8 (issue rate), k=256 and 1024 rounds (phase3q).
  Bundle checks confirm one op per round.
- Result: 2 cycles per round for add/sub/mul (f32, s32, u32), shifts, max, ceil/floor, conversions,
  pack/unpack. `cmp_select` (`where(v > c, v - c, v + c)`) takes 3 cycles per round, which is
  max(vcmp, vsub 2) + vsel 1. So vsel = 1 and vcmp <= 2. The compiler schedules mask consumers one
  bundle after a vcmp, and vcmp = 2 adds +9% to a calibration splash kernel with chunked masks, so
  the model keeps vcmp at 1.
- Unmeasured: logic ops (vand/vor/vxor), mask ops (vmand/vmor), vweird, vlaneseq; the bf16 select
  chain folded at compile time.

### F24. Scalar multiply (`smul.u32`) has a 2-cycle result latency
- Experiment: `pop_scalar` (phase3t): a chain of `s = s * c + d` (smul.u32 then sadd.s32) of n
  rounds, n = 64, 128, 256. The bundle dump shows the compiler issuing each round in 2 bundles.
- Result: 3.0 cycles per round (slope over n; intercepts agree for MXU and XLU variants). With sadd
  at 1, smul is 2.
- Confidence: high for the smul -> sadd pair. Other scalar ops are assumed to take 1 cycle.

### F25. Programs over ~64k bundles run from instruction overlays
- Experiment: VALU loops and straight-line code with 8k-96k bundles under sync-flag tracing
  (`TRACE_COMPUTE_AND_SYNC`, phase3s/phase3t). The trace has "sequencer_overlay" (flag 511)
  SyncWait / SyncNoWait / Set events.
- Results:
  - Up to 64,011 bundles: no overlay events inside the kernel, and the replay is exact.
  - 72,011 bundles and above: overlay events. Straight-line code switches every 14.67-14.70 us
    (about 21.7k bundles plus the switch). A SyncNoWait + Set pair marks entering a prefetched
    chunk and starting the next chunk's load.
  - A loop over more than two chunks waits at every wrap. The wait equals the load time minus the
    time spent in the last chunk.
  - A load takes 7.7-7.96 us, measured from Set to the end of the blocking wait. A 17-bundle
    chunk loads as slowly as a full one. Under philox's output writes a load takes 8.7 us.
  - Each switch costs about 0.27-0.3 us beyond the waits.
- Philox b1024 (85k bundles) shows the same pattern: 5 switches per grid step and an 8.2-8.7 us
  wait at every wrap, which is 13% of its time.
- Model (`model/overlay.py`):
  - A grid of 21,700-bundle chunks, also split at forward-branch targets in another chunk (philox
    has a fifth chunk at its pipeline-skip region).
  - 2 slots, LRU eviction, prefetch of the next chunk (after the last, the loop-head chunk).
  - Static branch prediction: a not-taken backward branch waits for its target chunk; a forward
    branch waits only when taken.
  - Fitted on the microbenchmarks to within 0.5%; philox (not fitted) goes from -15.4% to -1.0%.
- Confidence: high for the threshold, cadence, load time and wrap waits. Chunk size in bundles
  (rather than bytes), the slot count and the branch rule are inferred from these kernels.

- Capacity (phase3y): kernels of 64,010 bundles never overlay, and 65,408 or more always do. The
  launching module's TLP program is 341 bundles, so kernel + TLP fits exactly when it is at most
  65,536, matching a third-party claim of a 65,536-bundle instruction memory. The kernel's share is
  modeled as 65,195. (65,536 - 341) / 3 = 21,732, the fitted chunk size, which suggests three overlay
  slots of a third each. Unmodeled: at 65.4-66k bundles the third chunk is not prefetched and waits a
  full load (7.6 us); the model is -7% there.

### F26. Sync-flag tracing exposes every DMA wait and overlay load without changing kernel time
- `ProfileOptions.advanced_configuration={"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}`: device times
  match untraced runs. The "Tensor Core Sync Flag" line has SyncWait (blocked, with duration) or
  SyncNoWait per dma.done.wait, plus Set/Add events (`scripts/sync_diff.py`).

### F27. A scalar load's result is usable 6 cycles after the load
- Experiments: `pop_loop` inner loops (phase3w) whose step uses an in-loop `sld` 3, 4 or 5 bundles
  after it; `sld_unrolled` dependent pointer chases (phase3x), straight-line and unrolled 4 or 16 per
  loop iteration.
- Results:
  - The loops take +3, +2 and +1 cycles over their bundles, which is 6 minus the use distance.
  - The chases cost 6.0-6.25 cycles per load, although the compiler packs them 1-2 bundles apart.
  - With sld = 6 the model fits all of them within 3%.
  - A 2-cycle address lag on top (H11) overshoots the chases by 25-30%, so H11's lag is refuted.
- Open: `sld_chain` (F17) still takes one cycle per iteration more than the model.

### F28. Decoupled issue: scalar bundles run ahead of vector bundles waiting on a pop
- Experiment: `pop_loop` (phase3u): a pop, m dependent vadds, then an inner scalar loop of n trips.
- Results:
  - An 8-trip loop costs nothing on top of a 178-cycle MXU pop chain; in-order issue would be +37%.
  - 8 queued vector ops stay hidden; 16 stall the front end.
  - The model (`vector_queue` 12, range 11-13) fits within 0.5% except m=16 (+6%).
- Replaces H17. Calibration paged attention goes from +5.5% to +2%.

### F29. Strided DMAs lose ~0.65 ns per contiguous piece when pieces are far apart in HBM
- Experiments: `dma_strided` (phase3b: (8,128) f32 tiles of 4 KiB at stride ratios 2, 8 and 32) and
  `dma_mxu` (phase3z: 256x256 bf16 tiles, 4 KiB pieces at ratio 16, with and without MXU or VALU
  work). The descriptor's src/dst_stride give the piece size and the ratio.
- Results:
  - With pieces 8-32 apart, copies take ~0.5-0.8 ns more per piece than the same bytes contiguous.
  - With pieces 2 apart, there is no extra cost.
  - MXU work does not slow tile reads: with no compute, the stream is already 7-10% slower than the
    contiguous model.
- Model: 450 bytes of extra transfer per piece at ratio >= 8. Ratio 4 is unmeasured; calibration
  `add` (ratio 4, 16 KiB pieces) says it is free.
- Calibration matmul_tpu goes from -6% to -2.5% (worst -10.6% -> -8.3%). Pipelined two-input tile
  streams are still about -5% with no compute (open).
- Separately, the VMEM address parser now strips `ss:` stride fields. Strided vld/vst were missing
  their store->load wait (F20); vmem_access went from -35..-42% to within 0.2%.

## Hypotheses (not yet tested; the model must not depend on these)

- H2. MXU is 128x128.
- H3. Compiler per-bundle slot capacities, from `*-final_hlo-static-per-bundle-utilization.txt`:
  MXU 4, XLU 3, VALU 4, VPOP 2, EUP 1, VLOAD 3, VLOAD:FILL 3, VSTORE 1, VSTORE:SPILL 1, SALU 2.
  This is the compiler's model, not a timing measurement.
- H4. The Colab host has four v5e chips and the VM sees one (`TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`,
  `TPU_HOST_BOUNDS=1,1,1`). Unknown whether neighbours share anything that affects timing.
- H5. Number of DMA engines and transfers in flight. (Random addresses do not slow 4-64 KiB copies:
  `dma_gather` random vs sequential within 1% at 1, 4 and 16 in flight, phase 3e.)
- H8. XLU occupancy per op (fixed at 1 cycle: the nv=8, 16 and 32 chains are all latency-bound,
  108-110 cycles per round, and cannot identify it; the replay is 4-9% slow on the wide ones).
- H11. Mechanism of F17. Two fit sld_chain exactly: an `sld` result latency of 7 cycles, or a 2-cycle
  lag before a freshly written scalar register can be an sld address. On `dma_gather` (16 in flight)
  the first leaves -7%, the second -14% (none: -18%). Both make the calibration corpus worse
  (within 5%: 37/47 without, 22/47 with latency 7, 35/47 with the address lag), mostly paged
  attention, which is already over-predicted without them. Needs a benchmark that separates
  load-to-use from address dependence.
- H12. Vector issue: per-slot queues (pop, push, mxu, xlu, vld, vst, eup, valu) with the front end at
  most 8 cycles ahead (`issue_lead`). Fitted on core microbenchmarks (every one within 2.9%; strictly
  in-order issue leaves the 8-32-wide XLU chains 4-9% slow); the queue structure is inferred.
- H13. Scalar divide (`sdivrem` -> `spop.drf`) has no stall beyond the compiler's spacing. Unmeasured.
- H14 (sharpened with sync-flag traces, diag_sync): in a 32-row write-only pipeline a 128 KiB write must
  complete within ~600 cycles of issue (step 319 cycles, two buffers), while the start-then-wait
  per-copy time is 713 cycles; splitting write latency into head and tail (completion after the data)
  does not change this, because consecutive writes already overlap their start-up in the replay.
  Write waits stall 10-20 ns on hardware vs 80-100 ns in the replay in every small-block kernel traced
  (layernorm/softmax b8, matmul 256-tiles), and read waits stall more on hardware (300 vs 250 ns).
- H14. Pipelined writes complete faster than start-then-wait writes imply (32-row write-only stream:
  <= ~426 ns issue-to-done vs ~560 ns per sequential copy; replay +21%), and 1-read 1-write streams
  with moderate core work run slower than the replay (-17..-21%). Tested and rejected (fit does not
  improve them or breaks other DMA benchmarks): one shared queue, posted write completion,
  per-direction start-up adjustment, semaphore wake-up cost paid only on blocking waits.
- H15. Sublane-shuffle and masked VMEM accesses (`vld.sshfl`, `vst.sshfl`, `vst.msk`) hold their issue
  slot ~2.3 cycles. Fitted on the 4 calibration-split ragged paged attention records (no microbenchmark
  emits these ops: f32 and bf16 masked/strided Pallas accesses, and ragged's own strided uint32 K/V
  load in isolation, all compile to plain vld/vst, which cost what the model says; phase3k-3o2).
  Model parameter from the calibration corpus, not a measured fact.
- H16. A read's start-up does not progress while write data is moving (replaces a fixed mixed-direction
  start-up cost). With it, 1-read 1-write 32-row streams with 128-512 vadds/step go from -20% to
  -5..+3% and the overall DMA microbenchmark fit is unchanged (98-99/123 within 5%); a free refit
  wants a negative start-up (unphysical), so mix is held at 0. Calibration family biases move toward 0
  (layernorm -9% -> +2%, softmax -9% -> +3%, gmm -4% -> -1%). Inferred structure.
  With 512 KiB-1 MiB blocks it looks too strong: a read issued just after a 1 MiB write waits the
  whole 1.65 us of write data, so rope (1 MiB blocks) is +6..13% and dma_compute r1w1 b128 valu512
  is +6.4%. The 2 MiB-block streams in phase3u show the pause is capped. With a 400 ns cap, 37 of 51
  DMA microbenchmark and calibration streaming records are within 2% (26 without); calibration rope
  goes from +12.8% to +0.7% and RMSNorm b32 from +5.3% to -2.2%. The model uses `pause_cap_ns` 400.
  Side effect: tiled matmul kernels (two strided input streams) become 1-5 points more negative.
  Their tile streams were already too fast in the model (dma_mxu with no work: -5%).
- H17. Decoupled issue: scalar-only bundles keep issuing while up to Q vector bundles wait on a pop
  (`vector_queue`, off). In paged decode a 125-cycle MXU pop is followed by ~340 scalar bundles, and
  hardware is ~85 cycles per iteration faster than in-order issue. Q >= 16 brings calibration paged
  from +5.5% to +1.3%. `pop_scalar` could not test it because the compiler predicated its cond;
  phase3u's `pop_loop` uses a real inner loop.
- H18 (open): splash/flash tile-size bias. The `attn_step` microbenchmark (phase4a/4b) reproduces it:
  - Over-predicted +0.7% at 128x128 tiles, rising to +10% at 1024x256.
  - QK^T and PV alone fit within 2%. Adding the row max makes it +6-7% at large tiles; the softmax
    variant is +8%.
  - Single cross-lane reductions are right: 110 cycles/round for chains of max or sum
    (phase4b `vop_chain2` xmax/xsum, model 107), and 8 independent chains overlap fully.
  - Ruled out, as single-parameter changes:
    - a shorter sum-reduction latency (no effect);
    - a per-slot run-ahead for pops (no effect);
    - deeper vector queues (no effect);
    - MXU result latency or interval (breaks mxu_dot);
    - a global issue lead (breaks mxu_dot);
    - a 90-cycle max-reduction latency (fixes attention but breaks the measured chains).
  - In this context hardware is ~100 cycles per 1024-row step faster along MXU pop -> elementwise
    max -> cross-lane max -> pop.
- H10. DMA queueing structure: one FIFO per direction, heads interleaving (`model/dma.py:DirFifoHbm`).
  Chosen because it fits F14/F15 better than processor sharing or one FIFO; not observed directly.
- H6. Meaning of each bundle mnemonic and slot.
- H7. `add_one` slow outliers (12 to 16% of reps, up to 1.43x the median, spread through the
  run) come from something outside the kernel (refresh, host interference). Untested.
