"""Write the batch JSON files under batches/. Rerun after editing; batches are data."""

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
