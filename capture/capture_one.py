"""Capture one kernel config in a fresh process and write one record.

The parent (run_batch.py) sets LIBTPU_INIT_ARGS and XLA_FLAGS before this
process starts; jax must not be imported before they are in os.environ.

usage: capture_one.py CONFIG_JSON RECORD_DIR
"""

import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import sys
import time

if "jax" in sys.modules:
    raise SystemExit("jax imported before flags were set")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402

from capture import xplane  # noqa: E402
from corpus import kernels  # noqa: E402

PACKAGES = ("jax", "jaxlib", "libtpu", "xprof", "numpy")

KEEP = re.compile(
    r"final_bundles|final_hlo-static-per-bundle-utilization|static_llo_profile"
    r"|mosaic-dump-.*-(original|post-finalize-llo)\.txt|after_optimizations\.txt$"
)


def prune_dump(root):
    """Delete per-pass intermediates after they are listed in the record; keep what the model reads."""
    for dirpath, _, names in os.walk(root):
        for name in names:
            if not KEEP.search(name):
                os.remove(os.path.join(dirpath, name))


def versions():
    out = {p: importlib.metadata.version(p) for p in PACKAGES}
    out["python"] = platform.python_version()
    return out


def device_info(d):
    info = {"id": d.id, "platform": d.platform, "device_kind": d.device_kind, "str": str(d)}
    for attr in ("coords", "core_on_chip", "num_cores"):
        if hasattr(d, attr):
            info[attr] = getattr(d, attr)
    return info


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def machine(devices, client):
    return {
        "hostname": socket.gethostname(),
        "session": os.environ.get("TPUM_SESSION"),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cpu_model": cpu_model(),
        "platform_version": getattr(client, "platform_version", None),
        "devices": [device_info(d) for d in devices],
        "tpu_env": {k: v for k, v in os.environ.items() if k.startswith("TPU_")},
    }


def list_dump(root):
    files = []
    for dirpath, _, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(dirpath, name)
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            files.append(
                {"path": os.path.relpath(path, root), "bytes": os.path.getsize(path), "sha256": digest}
            )
    return sorted(files, key=lambda f: f["path"])


def compile_aot(fn, specs, topology_name):
    from jax.experimental import topologies

    topo = topologies.get_topology_desc(topology_name, "tpu")
    sharding = jax.sharding.SingleDeviceSharding(topo.devices[0])
    specs = [jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sharding) for s in specs]
    compiled = jax.jit(fn).lower(*specs).compile()
    return compiled, topo.devices, None


def make_inputs(specs):
    keys = jax.random.split(jax.random.key(0), len(specs))
    return [jax.random.normal(k, s.shape, s.dtype) for k, s in zip(keys, specs)]


def time_on_device(compiled, args, cfg, logdir):
    for _ in range(cfg["warmup"]):
        compiled(*args).block_until_ready()
    with jax.profiler.trace(logdir):
        for _ in range(cfg["reps"]):
            compiled(*args).block_until_ready()
    path = xplane.find_xplane(logdir)
    profile = xplane.load(path)
    by_line = xplane.kernel_durations(profile, cfg["kernel_tag"])
    return {
        "xplane_path": path,
        "kernel_tag": cfg["kernel_tag"],
        "reps": cfg["reps"],
        "warmup": cfg["warmup"],
        "by_line": {line: {"stats": xplane.stats(d), "samples_ns": d} for line, d in by_line.items()},
        "profile_summary": xplane.summarize(profile),
    }


def main(config_path, record_dir):
    with open(config_path) as f:
        cfg = json.load(f)
    t0 = time.time()
    fn, specs, kernel_class = kernels.build(cfg["kernel"], cfg["params"])

    if cfg["mode"] == "aot":
        compiled, devices, client = compile_aot(fn, specs, cfg["topology"])
        timing = None
    else:
        devices = jax.devices()
        client = devices[0].client
        args = make_inputs(specs)
        compiled = jax.jit(fn).lower(*args).compile()
        timing = time_on_device(compiled, args, cfg, os.path.join(record_dir, "profile"))

    try:
        cost = compiled.cost_analysis()
    except Exception as e:  # cost_analysis is unsupported on some backends
        cost = {"error": repr(e)}

    record = {
        "schema": 1,
        "record_id": cfg["record_id"],
        "config": cfg,
        "kernel": cfg["kernel"],
        "kernel_class": kernel_class,
        "params": cfg["params"],
        "arg_specs": [{"shape": list(s.shape), "dtype": str(s.dtype)} for s in specs],
        "versions": versions(),
        "flags": {k: os.environ.get(k) for k in ("LIBTPU_INIT_ARGS", "XLA_FLAGS")},
        "machine": machine(devices, client),
        "cost_analysis": cost,
        "dump_root": os.path.join(record_dir, "dump"),
        "dump_files": list_dump(os.path.join(record_dir, "dump")),
        "dump_pruned": cfg.get("prune_dump", True),
        "timing": timing,
        "wall_s": time.time() - t0,
    }
    if record["dump_pruned"]:
        prune_dump(record["dump_root"])
    tmp = os.path.join(record_dir, "record.json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, default=str)
    os.replace(tmp, os.path.join(record_dir, "record.json"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
