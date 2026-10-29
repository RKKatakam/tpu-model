# Bundle mnemonic catalog (libtpu 0.0.50.1, TPU v5 lite)

Status:
- **confirmed-functional**: the scalar interpreter (`model/machine.py`) implements it this way and
  replaying real kernels reproduces facts known from the kernel definition (DMA count per grid
  step, loop trip counts, no bounds-check halts). See `tests/test_machine.py`.
- **confirmed-timing**: a microbenchmark timed on device agrees (cited).
- **hypothesis**: inferred from names and context only.

## Syntax

| form | meaning | status |
|---|---|---|
| `addr [LB\|LH\|LE\|PB\|PF\|CT]: >... { op ;; op }` | one VLIW bundle; `>` per loop level; label key printed at file top | confirmed-functional |
| `/* Start region N */`, `/* Start/End empty region N */` | position of region N; `sbr.rel target = $regionN` jumps there | confirmed-functional |
| `op (%pN)` / `op (!%pN)` | op predicated on pN / not pN | confirmed-functional |
| `smov (%pN, %sOld), new` | conditional move: pN ? new : sOld | confirmed-functional |
| `/*name=*/operand` | operand label (DMA `hbm`, `size_in_granules`, `vmem`, `dst_syncflagno`, strides) | confirmed-functional |
| `[thread:$0]`, `[resolvable:$true]` | op attributes | hypothesis (`thread` = DMA issue thread) |
| `{}` | empty bundle (delay slot or latency padding) | confirmed-functional |
| `%sN_sK`, `%pN_pK`, `%vN_vK`, `%vmN_vmK` | SSA name + physical register sK/pK/vK/vmK | confirmed-functional |

## Scalar core

| mnemonic | semantics | status |
|---|---|---|
| `inlined_call_operand.{hbm,vmem,smem}` | binds operand `index: k` base address to a register at kernel entry, before bundle 0 | confirmed-functional |
| `smov x` | move | confirmed-functional |
| `sadd.s32 a,b`, `ssub.s32 a,b` (a-b), `smul` | 32-bit wraparound arithmetic | confirmed-functional (smul hypothesis) |
| `sand/sor/sxor.u32`, `sshll.u32 a,n`, `sshrl.u32`, `sshra.s32` | bitwise and shifts | confirmed-functional |
| `scmp.{eq,ne,lt,le,gt,ge}.{s32,u32}.totalorder a,b` | compare to predicate | confirmed-functional |
| `por/pand/pnand/pneg` | predicate logic | confirmed-functional |
| `scalar_select p,t,f` | p ? t : f | confirmed-functional |
| `scalar_lea.{hbm,vmem,sflag,smem} base,off` | base + off | confirmed-functional |
| `int_to_ptr.{hbm,vmem}` | free cast; reads its source after same-bundle writes | confirmed-functional |
| `sphi`, `vphi` | SSA phi marker, no runtime effect (moves are explicit `smov`/`vmov`) | confirmed-functional |
| `sld [smem:[a]]`, `sst [smem:[a]] v` | scalar memory load/store (spill slots) | confirmed-functional |
| `sbr.rel (p) target=$regionN` | branch; the next 4 bundles (delay slots) always issue | confirmed-functional |
| `shalt.err (p)` | trap if p (bounds checks, semaphore checks) | confirmed-functional |

## Sync flags and DMA

| mnemonic | semantics | status |
|---|---|---|
| `dma.hbm_to_vmem`, `dma.vmem_to_hbm` | async copy; `size_in_granules` x 32 B; adds granules to `dst_syncflagno` when done | confirmed-functional (32 B granule: 4096 B = 128 granules in add_one) |
| `dma.done.wait f, n` | stall until flag f >= n | confirmed-functional; timing: Phase 1b |
| `vsyncadd f, d` | f += d (u32; used with -n after a wait) | confirmed-functional |
| `vsyncpa f, v` | set flag f = v (0 at entry, 1 at exit) | hypothesis |
| `vsyncmov f` + `vpop.sfrf` | read flag f into the scalar side (end-of-kernel semaphore check) | confirmed-functional |

DMA descriptor comment fields: `base_bounds`, `dynamic_base_bounds`, `window_bounds`,
`iteration_bounds`, `strides`, `pad_low`, `pad_high`, `element_size_in_bytes`,
`second_minor_factor`; strided copies also carry `src_stride`, `dst_stride`,
`steps_per_stride` operands. Meaning of each: hypothesis.

## Vector unit and MXU (timing only; values not simulated)

| mnemonic | meaning | status |
|---|---|---|
| `vld [vmem:[a] sm:$mask]`, `vst` | vector load/store, sublane mask | hypothesis |
| `vadd.f32`, `vor.u32`, `vsel`, `vmov`, `vcmask`, `vrot.slane`, `vpack.c.f32.bf16` | VPU ALU / XLU ops | hypothesis; 4 vadd per bundle observed (valu_straight nv=8) |
| `vmatpush.bf16.{msra,msrb}.mxuK` | push weights into MXU K (double-buffered A/B) | hypothesis |
| `vmatmul.bf16[.vlgmr.msra].gmra.mxuK` | stream LHS rows through MXU K | hypothesis |
| `vpop.f32.mrf.mxuK` | pop a result vreg from MXU K's result FIFO | hypothesis |
