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


BATCHES = {
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

for batch_id, batch in BATCHES.items():
    with open(os.path.join(HERE, "batches", f"{batch_id}.json"), "w") as f:
        json.dump({"batch_id": batch_id, **batch}, f, indent=1)
    print(batch_id, len(batch["configs"]), "configs")
