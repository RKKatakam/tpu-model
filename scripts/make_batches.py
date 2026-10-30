"""Write the batch JSON files under batches/. Rerun after editing; batches are data."""

import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DUMP_FLAGS = {
    "libtpu": [
        "--xla_jf_dump_to={dump}/jf",
        "--xla_jf_dump_llo_text=true",
        "--xla_jf_dump_llo_static_gaps=true",
        "--xla_jf_emit_annotations=true",
        "--xla_jf_debug_level=2",
        "--xla_mosaic_dump_to={dump}/mosaic",
        "--xla_mosaic_enable_dump_debug_info=true",
        "--xla_mosaic_enable_llo_source_annotations=true",
        "--xla_jf_dump_static_llo_profile_to={dump}/static_profile",
    ],
    "xla": ["--xla_dump_to={dump}/xla", "--xla_dump_hlo_as_text"],
}

NOISE_KERNELS = [
    ("add_one", "k_add_one", {"rows": 8, "cols": 128}),
    ("add", "k_add", {"rows": 4096, "cols": 4096, "block_rows": 512, "block_cols": 1024}),
    ("col_sum", "k_col_sum", {"rows": 8192, "cols": 4096, "block_rows": 256}),
    ("matmul", "k_matmul", {"m": 2048, "n": 2048, "k": 2048, "bm": 512, "bn": 512, "bk": 512}),
]


def device_cfg(kernel, tag, params, repeat):
    return {
        "record_id": f"{kernel}__p{repeat}",
        "kernel": kernel,
        "kernel_tag": tag,
        "params": params,
        "mode": "device",
        "reps": 200,
        "warmup": 20,
    }


MINIMAL_FLAGS = {"libtpu": ["--xla_jf_dump_to={dump}/jf", "--xla_jf_dump_llo_text=true"], "xla": []}

ESTIMATE_FLAGS = {"libtpu": ["--xla_jf_dump_llo_critical_paths_to={dump}/critical_paths"]}


def aot_cfg(kernel, params, variant="full", **overrides):
    return {
        "record_id": f"{kernel}__aot__{variant}",
        "kernel": kernel,
        "params": params,
        "mode": "aot",
        "topology": "v5e:1x1",
        **overrides,
    }


def tag(kernel):
    return "k_" + kernel


def phase1_configs(mode):
    cfgs = []

    def add(kernel, params, rid):
        cfg = {"record_id": rid, "kernel": kernel, "kernel_tag": tag(kernel), "params": params}
        if mode == "aot":
            cfg |= {"mode": "aot", "topology": "v5e:1x1"}
        else:
            cfg |= {"mode": "device", "reps": 100, "warmup": 10}
        cfgs.append(cfg)

    for nv, ks in ((8, (0, 32, 128, 512, 1024, 2048, 4096)), (1, (0, 64, 256, 1024, 4096))):
        for k in ks:
            add("valu_straight", {"k": k, "nv": nv}, f"valu_straight__nv{nv}_k{k}")
    for k, trips in ((8, (1, 10, 100, 1000, 10000)), (1, (1000, 10000))):
        for trip in trips:
            add("valu_loop", {"trip": trip, "k": k, "nv": 8}, f"valu_loop__k{k}_t{trip}")
    for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        for reps in (1, 8):
            add("dma_seq", {"rows": rows, "reps": reps, "total_rows": 8192}, f"dma_seq__r{rows}_x{reps}")
    for bm, bn, bk in ((256, 256, 256), (512, 512, 512), (1024, 1024, 512), (256, 512, 1024)):
        p = {"m": 2048, "n": 2048, "k": 2048, "bm": bm, "bn": bn, "bk": bk}
        add("matmul", p, f"matmul__{bm}_{bn}_{bk}")
    return cfgs


def device(kernel, params, rid, reps=100, warmup=10):
    return {"record_id": rid, "kernel": kernel, "kernel_tag": tag(kernel), "params": params,
            "mode": "device", "reps": reps, "warmup": warmup}


def phase3a_configs():
    cfgs = []
    for op in ("add", "mul", "max", "exp", "roll_lane", "roll_sublane", "sum_lane", "cast"):
        for nv, k in ((1, 256), (1, 1024), (8, 128)):
            cfgs.append(device("vop_chain", {"k": k, "nv": nv, "op": op}, f"vop_chain__{op}_nv{nv}_k{k}"))
    for k in (64, 256):
        cfgs.append(device("mem_chain", {"k": k}, f"mem_chain__k{k}"))
    shapes = [(8, 128, 128), (16, 128, 128), (64, 128, 128), (256, 128, 128), (1024, 128, 128),
              (4096, 128, 128), (256, 256, 256), (256, 512, 512), (1024, 512, 512), (512, 1024, 1024)]
    for m, k, n in shapes:
        for reps in (4, 16):
            cfgs.append(device("mxu_dot", {"m": m, "k": k, "n": n, "reps": reps}, f"mxu_dot__{m}_{k}_{n}_x{reps}"))
    return cfgs


def phase3d_configs():
    """f32 matmuls (splash's p@v, paged): same method as the bf16 sweep in phase3a."""
    cfgs = []
    # k, n <= 256: the bf16 512-wide shapes spilled vregs (bench.checks), f32 doubles the pressure.
    shapes = [(8, 128, 128), (64, 128, 128), (256, 128, 128), (1024, 128, 128), (4096, 128, 128),
              (256, 256, 256), (1024, 256, 256)]
    for lhs, rhs in (("f32", "bf16"), ("f32", "f32")):
        for m, k, n in shapes:
            for reps in (4, 16):
                p = {"m": m, "k": k, "n": n, "reps": reps, "lhs": lhs, "rhs": rhs}
                cfgs.append(device("mxu_dot", p, f"mxu_dot__{lhs}x{rhs}_{m}_{k}_{n}_x{reps}"))
    return cfgs


def phase3e_configs():
    """Small DMAs at random vs sequential addresses with 1..16 in flight (paged attention)."""
    cfgs = []
    for rows, pages in ((8, 4096), (32, 4096), (128, 1024)):
        for order in ("seq", "random"):
            for inflight in (1, 4, 16):
                p = {"rows": rows, "pages": pages, "n": 512, "inflight": inflight, "order": order}
                cfgs.append(device("dma_gather", p, f"dma_gather__r{rows}_{order}_c{inflight}", reps=50))
    return cfgs


def phase3f_configs():
    """Core-side costs the replay misses (sld latency, DMA start+wait), XLU occupancy (H8) and
    transposed weight pushes (H9)."""
    cfgs = []
    for n in (256, 4096):
        cfgs.append(device("sld_chain", {"n": n, "size": 1024}, f"sld_chain__n{n}"))
    for n in (64, 256):
        for dma in (True, False):
            cfgs.append(device("dma_overhead", {"n": n, "k": 512, "dma": dma}, f"dma_overhead__n{n}_{'dma' if dma else 'nodma'}"))
    for op in ("roll_lane", "sum_lane"):
        for nv, k in ((16, 64), (32, 32)):
            cfgs.append(device("vop_chain", {"k": k, "nv": nv, "op": op}, f"vop_chain__{op}_nv{nv}_k{k}"))
    for m, k, n in ((256, 128, 128), (1024, 128, 128), (1024, 256, 256)):
        for reps in (4, 16):
            p = {"m": m, "k": k, "n": n, "reps": reps, "rhs_t": True}
            cfgs.append(device("mxu_dot", p, f"mxu_dot__bf16xbf16T_{m}_{k}_{n}_x{reps}"))
    return cfgs


def phase3g_configs():
    """Unit hand-off chains (paged/splash), DMA+compute overlap (small-block streaming), and sld
    load-to-use without an address dependence (H11)."""
    cfgs = []
    for dtype in ("f32", "bf16"):
        for kind in ("mxu", "mxu_xlu", "mxu_xlu_eup"):
            cfgs.append(device("unit_chain", {"k": 64, "kind": kind, "dtype": dtype}, f"unit_chain__{kind}_{dtype}_k64"))
    cfgs.append(device("unit_chain", {"k": 64, "kind": "xlu_eup", "dtype": "f32"}, "unit_chain__xlu_eup_f32_k64"))
    cfgs.append(device("unit_chain", {"k": 16, "kind": "mxu_xlu_eup", "dtype": "f32"}, "unit_chain__mxu_xlu_eup_f32_k16"))
    for br in (32, 128):
        for kind, amount in (("none", 0), ("valu", 256), ("valu", 1024), ("vmem", 32), ("vmem", 128)):
            cfgs.append(device("dma_compute", {"block_rows": br, "kind": kind, "amount": amount},
                               f"dma_compute__b{br}_{kind}{amount}"))
    for n in (256, 4096):
        cfgs.append(device("sld_use", {"n": n, "size": 1024}, f"sld_use__n{n}"))
    return cfgs


def phase3h_configs():
    """DMA streams with per-step core work: which direction slows when the core computes, and does
    VMEM traffic contend with DMA (small-tile pipelines: matmul, tgmm, ragged, layernorm b8)."""
    cfgs = []
    for reads, writes in ((1, 0), (0, 1), (1, 1)):
        for br in (32, 128):
            for amount in (0, 128, 256, 512, 1024):
                p = {"block_rows": br, "kind": "valu" if amount else "none", "amount": amount, "reads": reads, "writes": writes}
                cfgs.append(device("dma_compute", p, f"dma_compute__r{reads}w{writes}_b{br}_valu{amount}"))
    for br in (32, 128):
        for amount in (32, 128):
            p = {"block_rows": br, "kind": "vmem", "amount": amount, "reads": 1, "writes": 1}
            cfgs.append(device("dma_compute", p, f"dma_compute__r1w1_b{br}_vmem{amount}"))
    return cfgs


def phase3i_configs():
    """Fresh-weight matmuls (load -> push path, masked pushes) and a VMEM store->load chain."""
    cfgs = []
    for m, k, n in ((8, 128, 128), (128, 128, 128), (256, 128, 128), (1024, 128, 128), (256, 256, 256), (256, 64, 128)):
        for reps in (8, 32):
            cfgs.append(device("mxu_fresh", {"m": m, "k": k, "n": n, "reps": reps}, f"mxu_fresh__{m}_{k}_{n}_x{reps}"))
    for n in (256, 2048):
        cfgs.append(device("vmem_chain", {"n": n}, f"vmem_chain__n{n}"))
    return cfgs


def phase3j_configs():
    """XLU transposes (tgmm, x.T) and masked weight pushes."""
    cfgs = []
    for dtype in ("f32", "bf16"):
        for nv, k in ((1, 16), (1, 64), (4, 16)):
            cfgs.append(device("xpose_chain", {"k": k, "nv": nv, "dtype": dtype}, f"xpose_chain__{dtype}_nv{nv}_k{k}"))
    for m, k, n in ((128, 128, 128), (256, 128, 128), (1024, 128, 128)):
        for reps in (8, 32):
            cfgs.append(device("mxu_masked", {"m": m, "k": k, "n": n, "reps": reps}, f"mxu_masked__{m}_{k}_{n}_x{reps}"))
    return cfgs


def phase3k_configs():
    """Sublane-shuffle and masked VMEM accesses (ragged paged attention)."""
    cfgs = []
    for form in ("plain", "stride2", "stride4", "stride2_store", "masked"):
        for rows in (64, 512):
            for reps in (16, 64):
                cfgs.append(device("vmem_access", {"rows": rows, "reps": reps, "form": form}, f"vmem_access__{form}_r{rows}_x{reps}"))
    return cfgs


def phase3l_configs():
    """Live (output-bound) masked / sublane-shuffle / sublane-rotate VMEM accesses (ragged)."""
    cfgs = []
    for form in ("vld", "vst_msk", "vld_sshfl", "vst_sshfl", "vrot_slane"):
        for reps in (8, 32):
            cfgs.append(device("vmem_ops", {"n": 32, "reps": reps, "form": form}, f"vmem_ops__{form}_n32_x{reps}"))
    return cfgs


def phase3m_configs():
    """bf16 (packed) masked / sublane-shuffle VMEM accesses: the forms ragged paged attention emits."""
    cfgs = []
    for form in ("vld", "vst_msk", "vld_sshfl", "vst_sshfl"):
        for reps in (8, 32):
            cfgs.append(device("vmem_ops", {"n": 32, "reps": reps, "form": form, "dtype": "bf16"}, f"vmem_ops__bf16_{form}_n32_x{reps}"))
    return cfgs


def phase3n_configs():
    """megablox tgmm's masked inner step vs the same matmul unmasked (masked weight pushes)."""
    cfgs = []
    for tm, tk, tn in ((128, 128, 128), (256, 256, 256), (512, 128, 128)):
        for masked in (True, False):
            for reps in (8, 32):
                p = {"tm": tm, "tk": tk, "tn": tn, "reps": reps, "masked": masked}
                cfgs.append(device("tgmm_tile", p, f"tgmm_tile__{tm}_{tk}_{tn}_{'msk' if masked else 'plain'}_x{reps}"))
    return cfgs


def phase3o_configs():
    """ragged paged attention's strided uint32 K/V load (vld.sshfl) at several strides."""
    cfgs = []
    for step in (1, 2, 8, 16):
        for reps in (8, 32):
            cfgs.append(device("kv_strided", {"step": step, "n": 16, "reps": reps}, f"kv_strided__s{step}_n16_x{reps}"))
    return cfgs


def phase3p_configs():
    """int8 matmuls (s8 x s8 -> s32, as vLLM's w8a8 quantized matmul) and f32 transposed weight
    pushes (w8a16), plain and transposed weights; fit on reps 4, check on reps 16."""
    cfgs = []
    shapes = [(32, 128, 128), (256, 128, 128), (1024, 128, 128), (256, 256, 256), (1024, 256, 256)]
    for lhs, rhs, rhs_t in (("s8", "s8", False), ("s8", "s8", True), ("f32", "f32", True)):
        for m, k, n in shapes:
            for reps in (4, 16):
                p = {"m": m, "k": k, "n": n, "reps": reps, "lhs": lhs, "rhs": rhs, "rhs_t": rhs_t}
                cfgs.append(device("mxu_dot", p, f"mxu_dot__{lhs}x{rhs}{'T' if rhs_t else ''}_{m}_{k}_{n}_x{reps}"))
    return cfgs


MICRO_KERNELS = {"dma_compute", "dma_stream", "dma_inout"}
DIAG_SYNC_IDS = [
    # calibration-split kernels outside 5% (diagnosis never uses validation records)
    "layernorm__r16384_c4096_b8", "softmax__r16384_c4096_b8", "matmul__n1024_256_256_256",
    "matmul_tpu__2048_2048_2048_256_256_256", "matmul_tpu__4096_1024_4096_256_256_256", "matmul_tpu__4096_1024_4096_512_512_512",
    "paged__b16_pps64_pg32_cb4", "paged__b32_pps32_pg16_cb8", "paged__b8_pps64_pg16_cb4",
    "rope__t16384_h32_b128", "rope__t8192_h8_b256", "rmsnorm__r16384_c2048_b32", "prng__philox_b1024x1024",
    "tgmm__m8192_k512_n2048_g16_t256_256_256",
    # DMA microbenchmarks the DMA model still misses
    "dma_compute__r0w1_b32_valu0", "dma_compute__r0w1_b32_valu128", "dma_stream__r1w2_b32", "dma_inout__r128_x5",
]


def diag_sync_configs():
    """Copies of the configs above with hardware sync-flag tracing (TRACE_COMPUTE_AND_SYNC): every
    dma.done.wait's stall, for scripts/sync_diff.py. Diagnostic records, never scored."""
    import glob as _glob
    by_id = {}
    for f in sorted(_glob.glob(os.path.join(HERE, "batches", "*.json"))):
        for c in json.load(open(f))["configs"]:
            by_id.setdefault(c["record_id"], c)
    import sys as _sys
    _sys.path.insert(0, HERE)
    from corpus.split import split_of
    cfgs = []
    for rid in DIAG_SYNC_IDS:
        c = dict(by_id[rid])
        assert c["kernel"] in MICRO_KERNELS or split_of(rid) == "calib", rid
        c["record_id"] = "diag__" + rid
        c["reps"] = min(c.get("reps", 50), 20)
        c["profiler_options"] = {"advanced_configuration": {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}}
        cfgs.append(c)
    return cfgs


def phase3q_configs():
    """Latency (nv=1) and issue rate (nv=8) of vector ops the model has not measured (integer ALU,
    integer multiply, select / compare, conversions, divide)."""
    cfgs = []
    for op in ("add_s32", "sub_f32", "and_u32", "xor_u32", "shrl_u32", "shll_u32", "mul_u32", "mul_s32",
               "select", "cmp_select", "cvt_f32_s32", "pack_bf16", "div_f32"):
        for nv, k in ((1, 256), (1, 1024), (8, 128)):
            cfgs.append(device("vop_chain2", {"k": k, "nv": nv, "op": op}, f"vop_chain2__{op}_nv{nv}_k{k}"))
    return cfgs


def phase3r_configs():
    """Decode-shaped attention scores: few query rows against fresh, transposed K tiles (paged decode's
    push -> short matmul pattern)."""
    cfgs = []
    for lhs in ("f32", "bf16"):
        for m in (8, 32):
            for n in (128, 256, 512):
                for reps in (8, 32):
                    p = {"m": m, "k": 128, "n": n, "reps": reps, "lhs": lhs, "rhs_t": True}
                    cfgs.append(device("mxu_fresh", p, f"mxu_fresh__{lhs}xbf16T_{m}_128_{n}_x{reps}"))
    return cfgs


def phase3s_configs():
    """Program size vs instruction memory: VALU loops whose body is ~8k-96k bundles (nv=8 vadd rounds
    take 2 bundles each), with sync-flag tracing to see sequencer_overlay loads."""
    trace = {"advanced_configuration": {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}}
    cfgs = []
    runs = [("valu_loop", 4, k) for k in (4000, 8000, 12000, 16000, 20000, 24000, 32000, 48000)]
    runs += [("valu_loop", 1, k) for k in (16000, 48000)] + [("valu_straight", None, k) for k in (16000, 48000)]
    for kernel, trip, k in runs:
        p = {"k": k, "nv": 8} | ({"trip": trip} if trip else {})
        c = device(kernel, p, f"{kernel}__imem_k{k}" + (f"_t{trip}" if trip else ""), reps=20, warmup=3)
        c["profiler_options"] = trace
        c["timeout_s"] = 2400
        cfgs.append(c)
    return cfgs


def phase3t_configs():
    """phase3s sizes lost to a false session-gone stop, finer sizes between 40k (no overlays) and
    85k bundles (philox overlays), and pop -> scalar-region overlap (pop_scalar)."""
    trace = {"advanced_configuration": {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}}
    cfgs = []
    runs = [("valu_loop", 4, k) for k in (22000, 24000, 28000, 32000, 36000, 48000)]
    runs += [("valu_loop", 1, k) for k in (16000, 48000)] + [("valu_straight", None, k) for k in (16000, 48000)]
    for kernel, trip, k in runs:
        p = {"k": k, "nv": 8} | ({"trip": trip} if trip else {})
        c = device(kernel, p, f"{kernel}__imem_k{k}" + (f"_t{trip}" if trip else ""), reps=20, warmup=3)
        c["profiler_options"] = trace
        c["timeout_s"] = 2400
        cfgs.append(c)
    pops = [(kind, n, 0) for kind in ("mxu", "xlane") for n in (0, 64, 128, 256)] + [("mxu", 256, m) for m in (8, 16, 32)]
    for kind, n, m in pops:
        cfgs.append(device("pop_scalar", {"kind": kind, "n": n, "m": m, "trip": 64}, f"pop_scalar__{kind}_n{n}_m{m}"))
    return cfgs


def phase3u_configs():
    """Decoupled issue (pop_loop: a pop, `m` queued vector ops, then an inner scalar loop of n trips)
    and 1-read 1-write streams with 1-2 MiB blocks and per-step compute near the DMA time (rope's
    read waits)."""
    cfgs = []
    for kind in ("mxu", "xlane"):
        for n in (0, 8, 16, 32, 64):
            cfgs.append(device("pop_loop", {"kind": kind, "n": n, "m": 0, "trip": 64}, f"pop_loop__{kind}_n{n}_m0"))
    for m in (8, 16, 32, 64):
        cfgs.append(device("pop_loop", {"kind": "mxu", "n": 64, "m": m, "trip": 64}, f"pop_loop__mxu_n64_m{m}"))
    for br, amounts in ((256, (512, 1024, 2048, 3072)), (512, (1024, 2048, 3072, 4096, 6144))):
        for amount in amounts:
            p = {"block_rows": br, "kind": "valu", "amount": amount, "reads": 1, "writes": 1}
            cfgs.append(device("dma_compute", p, f"dma_compute__r1w1_b{br}_valu{amount}", reps=50))
    for reads, writes in ((1, 0), (0, 1)):
        p = {"block_rows": 512, "kind": "valu", "amount": 2048, "reads": reads, "writes": writes}
        cfgs.append(device("dma_compute", p, f"dma_compute__r{reads}w{writes}_b512_valu2048", reps=50))
    return cfgs


def phase3v_configs():
    """Ops of int8-quantizing kernels the model has not measured: one-vreg transposes (short vxpose),
    f32 <-> int8 packing, min, compare-equal select, round + clip."""
    cfgs = []
    for op in ("min_f32", "eq_select", "pack_s8", "round_clip"):
        for nv, k in ((1, 256), (1, 1024), (8, 128)):
            cfgs.append(device("vop_chain2", {"k": k, "nv": nv, "op": op}, f"vop_chain2__{op}_nv{nv}_k{k}"))
    for dtype, rows in (("bf16", 16), ("f32", 8)):
        for nv, k in ((1, 16), (1, 64), (4, 16)):
            cfgs.append(device("xpose_chain", {"k": k, "nv": nv, "dtype": dtype, "rows": rows},
                               f"xpose_chain__{dtype}_r{rows}_nv{nv}_k{k}"))
    return cfgs


def phase3w_configs():
    """What makes pop_loop's 7-bundle inner loop take 10 cycles: the in-loop sld feeding a multiply,
    an add or an xor, or a multiplier loaded before the loop."""
    cfgs = []
    for step in ("mul", "add", "xor", "mul_hoisted"):
        for n in (32, 64):
            p = {"kind": "xlane", "n": n, "m": 0, "trip": 64, "step": step}
            cfgs.append(device("pop_loop", p, f"pop_loop__xlane_n{n}_m0_{step}"))
    return cfgs


def phase3x_configs():
    """Scalar load-to-use latency away from branches: unrolled pointer chases, straight-line and in
    loops of 4 and 16 loads per iteration."""
    cfgs = []
    for n, unroll in ((64, 64), (256, 256), (256, 16), (1024, 16), (256, 4), (1024, 4)):
        cfgs.append(device("sld_unrolled", {"n": n, "unroll": unroll, "size": 64}, f"sld_unrolled__n{n}_u{unroll}"))
    return cfgs


def phase3y_configs():
    """Instruction memory capacity (third-party claim: 65536 bundles): straight-line VALU programs of
    2k + 9 bundles around 65536, traced for sequencer_overlay events."""
    trace = {"advanced_configuration": {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}}
    cfgs = []
    for k in (32700, 32755, 32760, 32763, 32764, 32770, 33000):
        c = device("valu_straight", {"k": k, "nv": 8}, f"valu_straight__imem_k{k}", reps=20, warmup=3)
        c["profiler_options"] = trace
        c["timeout_s"] = 2400
        cfgs.append(c)
    return cfgs


AOT_CHECK_IDS = [
    "splash__s2048_causal_q512_kv512_c512", "paged__b8_pps64_pg16_cb4", "matmul_tpu__4096_1024_4096_256_256_256",
    "tgmm__m8192_k512_n2048_g16_t256_256_256", "rope__t8192_h8_b256", "layernorm__r16384_c4096_b8",
    "ragged__decode_kv16_q32", "prng__philox_b1024x1024",
]


def aot_check_configs():
    """Deviceless compiles (topology v5e:1x1, CPU runtime) of measured calibration configs, to check
    that they give the same bundles as compiling on the TPU."""
    by_id = {}
    for f in sorted(glob.glob(os.path.join(HERE, "batches", "*.json"))):
        if os.path.basename(f).startswith("aot_check"):
            continue
        for c in json.load(open(f))["configs"]:
            by_id.setdefault(c["record_id"], c)
    out = []
    for rid in AOT_CHECK_IDS:
        c = dict(by_id[rid], mode="aot", topology="v5e:1x1", record_id="aot__" + rid)
        c.pop("profiler_options", None)
        out.append(c)
    return out


def phase3z_configs():
    """Do 2-D bf16 tile reads slow down under MXU work (tiled matmul read waits)? Same stream with
    no work, MXU matmuls, or VALU work of similar duration; one and two inputs."""
    cfgs = []
    base = {"rows": 4096, "cols": 4096, "bm": 256, "bn": 256}
    for n_in in (1, 2):
        runs = [("none", 0)] + [("mxu", a) for a in (1, 2, 4)] + [("valu", a) for a in (256, 512, 1024)]
        for work, amount in runs:
            p = base | {"n_in": n_in, "work": work, "amount": amount}
            cfgs.append(device("dma_mxu", p, f"dma_mxu__in{n_in}_{work}{amount}", reps=50))
    return cfgs


def phase4a_configs():
    """Flash-attention compute steps (attn_step) across tile sizes: splash is +4.8% at 256-row tiles
    and -2..-3% at 1024 in the replay."""
    cfgs = []
    for bq in (128, 256, 512, 1024):
        for bkv in (128, 256, 512):
            for reps in (4, 16):
                cfgs.append(device("attn_step", {"bq": bq, "bkv": bkv, "reps": reps}, f"attn_step__q{bq}_kv{bkv}_x{reps}"))
    return cfgs


def phase4b_configs():
    """Where the attention-step overprediction comes from: parts of the step (QK^T, + max, + softmax,
    PV alone) at small and large tiles, and max / sum cross-lane reduction chains (nv=1 latency, nv=8
    throughput)."""
    cfgs = []
    for part in ("qk", "qk_max", "qk_softmax", "pv"):
        for bq, bkv in ((128, 128), (512, 256), (1024, 256), (1024, 512)):
            for reps in (4, 16):
                cfgs.append(device("attn_step", {"bq": bq, "bkv": bkv, "reps": reps, "part": part},
                                   f"attn_step__{part}_q{bq}_kv{bkv}_x{reps}"))
    for op in ("xmax", "xsum"):
        for nv, k in ((1, 64), (1, 256), (8, 64)):
            cfgs.append(device("vop_chain2", {"k": k, "nv": nv, "op": op}, f"vop_chain2__{op}_nv{nv}_k{k}"))
    return cfgs


def phase4c_configs():
    """H18: max reductions on VMEM-loaded scores (load_max), and QK^T / loaded scores reduced by
    elementwise column max only (qk_colmax, load_colmax): MXU -> XLU hand-off vs reduction itself."""
    cfgs = []
    for part in ("load_max", "qk_colmax", "load_colmax"):
        for bq, bkv in ((128, 128), (512, 256), (1024, 256), (1024, 512)):
            for reps in (4, 16):
                cfgs.append(device("attn_step", {"bq": bq, "bkv": bkv, "reps": reps, "part": part},
                                   f"attn_step__{part}_q{bq}_kv{bkv}_x{reps}"))
    return cfgs


def phase4d_configs():
    """Concurrent strided tile reads (tiled matmul read waits) across tile sizes and matrix widths, and
    pipelined streams of small (32-256 KiB) blocks (narrow layernorm/softmax)."""
    cfgs = []
    for cols in (1024, 4096):
        for bm, bn in ((256, 256), (512, 512), (512, 1024), (1024, 512)):
            for n_in in (1, 2):
                p = {"rows": 4096, "cols": cols, "bm": bm, "bn": bn, "n_in": n_in, "work": "none", "amount": 0}
                cfgs.append(device("dma_mxu", p, f"dma_mxu__c{cols}_{bm}x{bn}_in{n_in}", reps=50))
    for br in (8, 16, 64):
        for nr, nw in ((1, 1), (1, 0), (0, 1)):
            p = {"rows": 16384, "cols": 1024, "block_rows": br, "reads": nr, "writes": nw}
            cfgs.append(device("dma_stream", p, f"dma_stream__c1024_r{nr}w{nw}_b{br}", reps=50))
    return cfgs


def phase4e_configs():
    """Per-row scale broadcast with and without the row -> column transpose (int8 activation
    quantization step)."""
    cfgs = []
    for form in ("row", "col"):
        for rows in (128, 256, 512):
            for cv in (1, 4):
                for reps in (8, 32):
                    p = {"rows": rows, "cols_vregs": cv, "reps": reps, "form": form}
                    cfgs.append(device("scale_bcast", p, f"scale_bcast__{form}_r{rows}_c{cv}_x{reps}"))
    return cfgs


def trace_probe_configs():
    """Does an in-kernel named_scope produce trace events, and under which profiler option?"""
    cfgs = []
    modes = [None, "TRACE_ONLY_XLA", "TRACE_COMPUTE", "TRACE_COMPUTE_AND_SYNC"]
    for mode in modes:
        cfg = device("scoped_stream", {"block_rows": 32, "amount": 256}, f"scoped_stream__{(mode or 'default').lower()}", reps=20)
        if mode:
            cfg["profiler_options"] = {"advanced_configuration": {"tpu_trace_mode": mode}}
        cfgs.append(cfg)
    cfgs.append(device("dma_compute", {"block_rows": 32, "kind": "valu", "amount": 256, "reads": 1, "writes": 1},
                       "dma_compute__r1w1_b32_valu256_unscoped", reps=20))
    return cfgs


def phase3b_configs():
    cfgs = []
    for rows in (1, 8, 64, 512, 2048, 8192):
        for reps in (1, 8):
            cfgs.append(device("dma_out", {"rows": rows, "reps": reps}, f"dma_out__r{rows}_x{reps}"))
    for rows in (8, 128, 1024, 2048):
        for streams in (2, 4, 8):
            for reps in (1, 5):
                cfgs.append(device("dma_conc", {"rows": rows, "streams": streams, "reps": reps},
                                   f"dma_conc__r{rows}_s{streams}_x{reps}"))
    for rows in (8, 128, 1024, 4096):
        for reps in (1, 5):
            cfgs.append(device("dma_inout", {"rows": rows, "reps": reps}, f"dma_inout__r{rows}_x{reps}"))
    for rows in (8, 256, 2048):
        for width in (256, 1024, 4096):
            for reps in (1, 8):
                cfgs.append(device("dma_strided", {"rows": rows, "width": width, "reps": reps},
                                   f"dma_strided__r{rows}_w{width}_x{reps}"))
    return cfgs


CORPUS_TAGS = {
    "flash": "_flash_attention_kernel",
    "splash": "splash_mha_fwd",
    "paged": "paged_flash_attention_kernel",
    "gmm": "%kernel",
    "layernorm": "k_layernorm",
    "softmax": "k_softmax",
    "matmul": "k_matmul",
    "ragged": "ragged_paged_attention", "matmul_tpu": "matmul", "tgmm": "tgmm", "prng": "prng",
    "rmsnorm": "k_rmsnorm", "rope": "k_rope",
    "vllm_rpa": "rpa", "vllm_qmm": "qmm", "vllm_flash": "flash", "vllm_gmm": "gmm",
    "add": "k_add",
    "col_sum": "k_col_sum",
}


def phase3c_configs():
    cfgs = []
    for nr, nw in ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)):
        # 1024-row f32 blocks exceed the 16 MiB scoped VMEM once a read and a write are double-buffered.
        for br in (32, 128, 512):
            p = {"rows": 16384, "cols": 1024, "block_rows": br, "reads": nr, "writes": nw}
            cfgs.append(device("dma_stream", p, f"dma_stream__r{nr}w{nw}_b{br}", reps=50))
    return cfgs


def corpus(kernel, params, rid, reps=50):
    return {"record_id": rid, "kernel": kernel, "kernel_tag": CORPUS_TAGS[kernel], "params": params,
            "mode": "device", "reps": reps, "warmup": 5, "seed": 0, "timeout_s": 600}


def corpus_configs():
    """Autotune-style sweeps: each kernel at a few shapes x several tile configs."""
    cfgs = []
    for s_len, causal in ((2048, True), (4096, True), (2048, False)):
        for bq, bk in ((128, 128), (256, 256), (512, 512), (512, 1024), (1024, 512)):
            if bk > s_len or bq > s_len:
                continue
            p = {"b": 1, "h": 8, "s": s_len, "d": 128, "causal": causal, "block_q": bq,
                 "block_k_major": bk, "block_k": min(bk, 512)}
            cfgs.append(corpus("flash", p, f"flash__s{s_len}_c{int(causal)}_q{bq}_k{bk}"))
    for s_len, mask in ((2048, "causal"), (4096, "causal"), (4096, "local"), (2048, "full")):
        for bq, bkv, bkc in ((128, 128, 128), (256, 256, 256), (512, 512, 512), (512, 1024, 512), (1024, 1024, 512)):
            p = {"h": 8, "s": s_len, "d": 128, "mask": mask, "block_q": bq, "block_kv": bkv,
                 "block_kv_compute": bkc, "window": 512}
            cfgs.append(corpus("splash", p, f"splash__s{s_len}_{mask}_q{bq}_kv{bkv}_c{bkc}"))
    for batch, pps, page in ((8, 64, 16), (32, 32, 16), (16, 64, 32)):
        for ppcb in (4, 8, 16):
            p = {"batch": batch, "q_heads": 32, "kv_heads": 8, "head_dim": 128, "page_size": page,
                 "pages_per_seq": pps, "total_pages": 4096, "pages_per_compute_block": ppcb}
            cfgs.append(corpus("paged", p, f"paged__b{batch}_pps{pps}_pg{page}_cb{ppcb}"))
    for m, k, n, g in ((4096, 1024, 1024, 8), (8192, 2048, 512, 16)):
        for tm, tk, tn in ((128, 128, 128), (256, 512, 512), (512, 512, 512), (512, 1024, 256)):
            if tk > k or tn > n:
                continue
            p = {"m": m, "k": k, "n": n, "groups": g, "tm": tm, "tk": tk, "tn": tn}
            cfgs.append(corpus("gmm", p, f"gmm__m{m}_k{k}_n{n}_g{g}_t{tm}_{tk}_{tn}"))
    for rows, cols in ((8192, 1024), (16384, 4096)):
        for br in (8, 64, 256, 1024):
            if br * cols * 4 * 4 > 12 * 2**20:
                continue
            for kname in ("layernorm", "softmax"):
                p = {"rows": rows, "cols": cols, "block_rows": br}
                cfgs.append(corpus(kname, p, f"{kname}__r{rows}_c{cols}_b{br}"))
    for n_, (bm, bn, bk) in ((4096, (512, 512, 512)), (4096, (1024, 1024, 512)), (4096, (256, 1024, 512)),
                             (1024, (256, 256, 256)), (1024, (512, 512, 1024)), (8192, (1024, 1024, 1024))):
        p = {"m": n_, "n": n_, "k": n_, "bm": bm, "bn": bn, "bk": bk}
        cfgs.append(corpus("matmul", p, f"matmul__n{n_}_{bm}_{bn}_{bk}"))
    for rows, cols, br, bc in ((8192, 2048, 256, 2048), (8192, 2048, 1024, 512), (2048, 2048, 64, 2048)):
        p = {"rows": rows, "cols": cols, "block_rows": br, "block_cols": bc}
        cfgs.append(corpus("add", p, f"add__r{rows}_c{cols}_b{br}x{bc}"))
    for rows, cols, br in ((16384, 2048, 128), (16384, 2048, 512), (4096, 8192, 64)):
        p = {"rows": rows, "cols": cols, "block_rows": br}
        cfgs.append(corpus("col_sum", p, f"col_sum__r{rows}_c{cols}_b{br}"))
    return cfgs


SPLASH_TILES = ((128, 128, 128), (256, 512, 256), (512, 512, 512), (512, 1024, 512), (1024, 1024, 1024))


def corpus2_configs():
    """Splash attention (the TPU production attention kernel) across head dim, length, MQA,
    mask type and logit soft cap; each line is one autotune sweep over SPLASH_TILES."""
    shapes = [
        {"h": 16, "s": 4096, "d": 64, "mask": "causal"},
        {"h": 4, "s": 2048, "d": 256, "mask": "causal"},
        {"h": 8, "s": 4096, "d": 128, "mask": "causal", "mqa": True},
        {"h": 8, "s": 8192, "d": 128, "mask": "causal"},
        {"h": 8, "s": 4096, "d": 128, "mask": "chunked", "window": 1024},
        {"h": 8, "s": 2048, "d": 128, "mask": "causal", "soft_cap": 30.0},
        {"h": 8, "s": 1024, "d": 128, "mask": "full"},
    ]
    cfgs = []
    for shape in shapes:
        for bq, bkv, bkc in SPLASH_TILES:
            if max(bq, bkv) > shape["s"]:
                continue
            p = shape | {"block_q": bq, "block_kv": bkv, "block_kv_compute": bkc}
            tag = "_".join(f"{k}{v}" for k, v in shape.items() if k not in ("h", "s", "d", "mask"))
            rid = f"splash__h{shape['h']}_s{shape['s']}_d{shape['d']}_{shape['mask']}{'_' + tag if tag else ''}_q{bq}_kv{bkv}_c{bkc}"
            cfgs.append(corpus("splash", p, rid.replace("True", "").replace(".0", "")))
    return cfgs


def corpus3_configs():
    """Kernels from other sources: ragged paged attention (vLLM), Pallas tiled matmul, megablox tgmm,
    threefry/philox PRNG, and hand-written RMSNorm and RoPE; each line is an autotune sweep."""
    cfgs = []
    base = {"q_heads": 32, "kv_heads": 8, "head_dim": 128, "page_size": 16, "total_pages": 4096}
    shapes = {"decode": {"n_decode": 32, "n_prefill": 0, "prefill_len": 0, "kv_len": 2048, "prefix": 0},
              "mixed": {"n_decode": 16, "n_prefill": 2, "prefill_len": 256, "kv_len": 1024, "prefix": 512},
              "prefill": {"n_decode": 0, "n_prefill": 4, "prefill_len": 512, "kv_len": 0, "prefix": 0}}
    for name, shp in shapes.items():
        for kvb, qb in ((8, 16), (16, 32), (32, 64), (16, 128)):
            cfgs.append(corpus("ragged", base | shp | {"kv_pages_per_block": kvb, "queries_per_block": qb},
                               f"ragged__{name}_kv{kvb}_q{qb}"))
    for m, k, n in ((2048, 2048, 2048), (4096, 1024, 4096)):
        for bm, bn, bk in ((256, 256, 256), (512, 512, 512), (1024, 1024, 256), (512, 1024, 512), (256, 512, 1024)):
            cfgs.append(corpus("matmul_tpu", {"m": m, "k": k, "n": n, "bm": bm, "bn": bn, "bk": bk},
                               f"matmul_tpu__{m}_{k}_{n}_{bm}_{bn}_{bk}"))
    for m, k, n, g in ((4096, 1024, 1024, 8), (8192, 512, 2048, 16)):
        for tm, tk, tn in ((128, 128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 1024)):
            cfgs.append(corpus("tgmm", {"m": m, "k": k, "n": n, "groups": g, "tm": tm, "tk": tk, "tn": tn},
                               f"tgmm__m{m}_k{k}_n{n}_g{g}_t{tm}_{tk}_{tn}"))
    for algo in ("threefry", "philox"):
        for bm, bn in ((256, 256), (512, 512), (256, 1024), (1024, 1024)):
            cfgs.append(corpus("prng", {"algo": algo, "rows": 4096, "cols": 4096, "block_rows": bm, "block_cols": bn},
                               f"prng__{algo}_b{bm}x{bn}"))
    for rows, cols, brs in ((8192, 4096, (16, 64)), (16384, 2048, (32, 128, 256))):
        for br in brs:
            cfgs.append(corpus("rmsnorm", {"rows": rows, "cols": cols, "block_rows": br}, f"rmsnorm__r{rows}_c{cols}_b{br}"))
    for tokens, heads, bts in ((8192, 8, (64, 256, 1024)), (16384, 32, (128, 512))):
        for bt in bts:
            cfgs.append(corpus("rope", {"tokens": tokens, "heads": heads, "head_dim": 128, "block_tokens": bt},
                               f"rope__t{tokens}_h{heads}_b{bt}"))
    return cfgs


def external_vllm_configs():
    """vllm-project/tpu-inference kernels (vendored; external test set, never used for calibration)."""
    cfgs = []
    base = {"q_heads": 32, "kv_heads": 8, "head_dim": 128, "page_size": 16, "num_pages": 4096}
    shapes = {"decode": {"n_decode": 32, "n_prefill": 0, "prefill_len": 0, "kv_len": 2048, "prefix": 0},
              "mixed": {"n_decode": 16, "n_prefill": 2, "prefill_len": 256, "kv_len": 1024, "prefix": 512},
              "prefill": {"n_decode": 0, "n_prefill": 4, "prefill_len": 512, "kv_len": 0, "prefix": 0}}
    for name, shp in shapes.items():
        for mb in ((32, 128, 32, 128), (64, 256, 32, 128), (128, 512, 64, 256)):
            # vLLM's default decode block (bkv 2048) needs 147 MB of VMEM on v5e; use the size its tests use
            p = base | shp | {"m_blocks": list(mb), "d_blocks": [1, 512, 1, 512], "vmem_limit_bytes": 100 * 2**20}
            cfgs.append(corpus("vllm_rpa", p, f"vllm_rpa__{name}_m{mb[0]}_{mb[1]}"))
    for mode in ("w8a8", "w8a16"):
        for m in (16, 128, 512, 2048):
            cfgs.append(corpus("vllm_qmm", {"m": m, "k": 4096, "n": 4096, "mode": mode}, f"vllm_qmm__{mode}_m{m}_k4096_n4096"))
    for s_len, causal in ((2048, True), (4096, False)):
        for bq, bk in ((256, 256), (512, 512), (512, 1024)):
            p = {"b": 1, "h": 8, "s": s_len, "d": 128, "causal": causal, "block_q": bq, "block_k_major": bk, "block_k": min(bk, 512)}
            cfgs.append(corpus("vllm_flash", p, f"vllm_flash__s{s_len}_c{int(causal)}_q{bq}_k{bk}"))
    for tm, tk, tn in ((128, 128, 128), (256, 512, 512), (512, 1024, 512)):
        p = {"m": 4096, "k": 2048, "n": 1024, "groups": 8, "tm": tm, "tk": tk, "tn": tn}
        cfgs.append(corpus("vllm_gmm", p, f"vllm_gmm__m4096_k2048_n1024_g8_t{tm}_{tk}_{tn}"))
    return cfgs


def split(cfgs, parts):
    return [cfgs[i::parts] for i in range(parts)]


BATCHES = {
    "phase3a": {"flags": DUMP_FLAGS, "configs": phase3a_configs()},
    "phase3b": {"flags": DUMP_FLAGS, "configs": phase3b_configs()},
    "phase3c": {"flags": DUMP_FLAGS, "configs": phase3c_configs()},
    "phase3d": {"flags": DUMP_FLAGS, "configs": phase3d_configs()},
    "phase3e": {"flags": DUMP_FLAGS, "configs": phase3e_configs()},
    "phase3f": {"flags": DUMP_FLAGS, "configs": phase3f_configs()},
    "phase3g": {"flags": DUMP_FLAGS, "configs": phase3g_configs()},
    "phase3h": {"flags": DUMP_FLAGS, "configs": phase3h_configs()},
    "phase3i": {"flags": DUMP_FLAGS, "configs": phase3i_configs()},
    "phase3j": {"flags": DUMP_FLAGS, "configs": phase3j_configs()},
    "phase3k": {"flags": DUMP_FLAGS, "configs": phase3k_configs()},
    "phase3l": {"flags": DUMP_FLAGS, "configs": phase3l_configs()},
    "phase3m": {"flags": DUMP_FLAGS, "configs": phase3m_configs()},
    "phase3n": {"flags": DUMP_FLAGS, "configs": phase3n_configs()},
    "phase3o": {"flags": DUMP_FLAGS, "configs": phase3o_configs()},
    "trace_probe": {"flags": DUMP_FLAGS, "configs": trace_probe_configs()},
    "phase3p": {"flags": DUMP_FLAGS, "configs": phase3p_configs()},
    "diag_sync": {"flags": DUMP_FLAGS, "configs": diag_sync_configs()},
    "phase3q": {"flags": DUMP_FLAGS, "configs": phase3q_configs()},
    "phase3r": {"flags": DUMP_FLAGS, "configs": phase3r_configs()},
    "phase3s": {"flags": DUMP_FLAGS, "configs": phase3s_configs()},
    "phase3t": {"flags": DUMP_FLAGS, "configs": phase3t_configs()},
    "phase3u": {"flags": DUMP_FLAGS, "configs": phase3u_configs()},
    "phase3v": {"flags": DUMP_FLAGS, "configs": phase3v_configs()},
    "phase3w": {"flags": DUMP_FLAGS, "configs": phase3w_configs()},
    "phase3x": {"flags": DUMP_FLAGS, "configs": phase3x_configs()},
    "phase3y": {"flags": DUMP_FLAGS, "configs": phase3y_configs()},
    "phase3z": {"flags": DUMP_FLAGS, "configs": phase3z_configs()},
    "phase4a": {"flags": DUMP_FLAGS, "configs": phase4a_configs()},
    "phase4b": {"flags": DUMP_FLAGS, "configs": phase4b_configs()},
    "phase4c": {"flags": DUMP_FLAGS, "configs": phase4c_configs()},
    "phase4d": {"flags": DUMP_FLAGS, "configs": phase4d_configs()},
    "phase4e": {"flags": DUMP_FLAGS, "configs": phase4e_configs()},
    "aot_check": {"flags": DUMP_FLAGS, "configs": aot_check_configs()},
    "phase3o2": {"flags": DUMP_FLAGS, "configs": [c for c in phase3o_configs() if c["params"]["step"] > 1]},
    "phase1_aot": {"flags": DUMP_FLAGS, "configs": phase1_configs("aot")},
    "phase1_device": {"flags": DUMP_FLAGS, "configs": phase1_configs("device")},
    "phase0_aot": {
        "probe": True,
        "flags": DUMP_FLAGS,
        "configs": [aot_cfg(k, p) for k, _, p in NOISE_KERNELS]
        + [aot_cfg(k, p, "estimate", extra_flags=ESTIMATE_FLAGS) for k, _, p in NOISE_KERNELS]
        + [aot_cfg(k, p, "minimal", flags=MINIMAL_FLAGS) for k, _, p in NOISE_KERNELS],
    },
    "phase0_noise": {
        "probe": True,
        "flags": DUMP_FLAGS,
        "configs": [device_cfg(k, t, p, r) for r in range(3) for k, t, p in NOISE_KERNELS],
    },
}

for i, part in enumerate(split(corpus_configs(), 2)):
    BATCHES[f"corpus1_part{i}"] = {"flags": DUMP_FLAGS, "configs": part}
# These aborted in corpus1 because XLA's static LLO profiler is fatal on their DMAs; rerun with the
# run_batch fallback. matmul__n8192_1024_1024_1024 is not rerun: it exceeds scoped VMEM (infeasible).
BATCHES["corpus1_rerun"] = {"flags": DUMP_FLAGS,
                            "configs": [c for c in corpus_configs() if c["kernel"] in ("splash", "paged", "gmm")]}

BATCHES["corpus2_splash"] = {"flags": DUMP_FLAGS, "configs": corpus2_configs()}
for i, part in enumerate(split(external_vllm_configs(), 2)):
    BATCHES[f"external_vllm_part{i}"] = {"flags": DUMP_FLAGS, "configs": part}
# part0's session was reclaimed before its non-attention records were downloaded
BATCHES["external_vllm_part0b"] = {"flags": DUMP_FLAGS, "configs": [c for c in split(external_vllm_configs(), 2)[0] if c["kernel"] != "vllm_rpa"]}
BATCHES["external_vllm_rpa"] = {"flags": DUMP_FLAGS, "configs": [c for c in external_vllm_configs() if c["kernel"] == "vllm_rpa"]}
for i, part in enumerate(split(corpus3_configs(), 2)):
    BATCHES[f"corpus3_part{i}"] = {"flags": DUMP_FLAGS, "configs": part}

for batch_id, batch in BATCHES.items():
    with open(os.path.join(HERE, "batches", f"{batch_id}.json"), "w") as f:
        json.dump({"batch_id": batch_id, **batch}, f, indent=1)
    print(batch_id, len(batch["configs"]), "configs")
