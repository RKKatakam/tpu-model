"""Colab-side batch launcher. Runs in the Colab kernel; never imports jax.

Builds a venv from requirements.lock, checks the installed versions against
requirements.in, then runs each config of the batch as a fresh capture_one.py
subprocess with its own LIBTPU_INIT_ARGS / XLA_FLAGS, one at a time. Configs
with an existing record.json are skipped, so a rerun resumes.

Inputs (env): TPUM_ROOT, TPUM_BATCH (batch json path), TPUM_SESSION.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = os.environ.get("TPUM_ROOT", "/content/tpum")
VENV = os.environ.get("TPUM_VENV", "/content/tpum_venv")
PY = os.path.join(VENV, "bin", "python")


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def ensure_venv():
    lock = os.path.join(ROOT, "requirements.lock")
    with open(lock, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    marker = os.path.join(VENV, ".lock_sha256")
    if os.path.exists(marker) and open(marker).read() == digest:
        return
    if shutil.which("uv") is None:
        sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
    sh([uv, "venv", "--clear", "--python", "3.12", VENV])
    sh([uv, "pip", "install", "--python", PY, "--require-hashes", "-r", lock])
    with open(marker, "w") as f:
        f.write(digest)


def check_versions():
    pins = {}
    with open(os.path.join(ROOT, "requirements.in")) as f:
        for line in f:
            if "==" in line:
                name, ver = line.strip().split("==")
                pins[name] = ver
    code = (
        "import importlib.metadata as m, json, platform;"
        f"print(json.dumps({{p: m.version(p) for p in {sorted(pins)!r}}} | {{'python': platform.python_version()}}))"
    )
    got = json.loads(subprocess.run([PY, "-c", code], check=True, capture_output=True, text=True).stdout)
    print("venv versions:", got, flush=True)
    bad = {p: (v, got.get(p)) for p, v in pins.items() if got.get(p) != v}
    if bad:
        raise SystemExit(f"pinned versions not installed (want, got): {bad}")


def flag_env(batch, cfg, dump):
    env = dict(os.environ)
    base = cfg.get("flags", batch["flags"])
    for key, var in (("libtpu", "LIBTPU_INIT_ARGS"), ("xla", "XLA_FLAGS")):
        flags = base.get(key, []) + cfg.get("extra_flags", {}).get(key, [])
        flags = [f.replace("{dump}", dump) for f in flags]
        env.pop(var, None)
        if flags:
            env[var] = " ".join(flags)
    if batch.get("cpu_only"):
        env["JAX_PLATFORMS"] = "cpu"
    return env


def run_one(batch, cfg, out_dir):
    record_dir = os.path.join(out_dir, cfg["record_id"])
    if os.path.exists(os.path.join(record_dir, "record.json")):
        print(f"skip {cfg['record_id']}", flush=True)
        return True
    shutil.rmtree(record_dir, ignore_errors=True)
    dump = os.path.join(record_dir, "dump")
    os.makedirs(dump)
    with open(os.path.join(record_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    env = flag_env(batch, cfg, dump)
    env["TPUM_SESSION"] = os.environ.get("TPUM_SESSION", "")
    cmd = [PY, os.path.join(ROOT, "capture", "capture_one.py"), os.path.join(record_dir, "config.json"), record_dir]
    with open(os.path.join(record_dir, "stdout.txt"), "w") as out, open(
        os.path.join(record_dir, "stderr.txt"), "w"
    ) as err:
        try:
            rc = subprocess.run(cmd, env=env, stdout=out, stderr=err, timeout=cfg.get("timeout_s", 900)).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    ok = rc == 0
    if not ok:
        with open(os.path.join(record_dir, "stderr.txt")) as f:
            tail = f.read()[-3000:]
        print(f"FAIL {cfg['record_id']} rc={rc}\n{tail}", flush=True)
    else:
        print(f"ok   {cfg['record_id']}", flush=True)
    return ok


def main():
    with open(os.path.join(ROOT, os.environ["TPUM_BATCH"])) as f:
        batch = json.load(f)
    session = os.environ["TPUM_SESSION"]
    ensure_venv()
    check_versions()
    out_dir = os.path.join(ROOT, "results", batch["batch_id"], session)
    os.makedirs(out_dir, exist_ok=True)
    if batch.get("probe"):
        with open(os.path.join(out_dir, "probe.txt"), "w") as f:
            subprocess.run([PY, os.path.join(ROOT, "capture", "probe_env.py")], stdout=f, stderr=subprocess.STDOUT)
    results = [run_one(batch, cfg, out_dir) for cfg in batch["configs"]]
    archive = os.path.join(ROOT, "results", f"{batch['batch_id']}__{session}.tgz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_dir, arcname=os.path.join(batch["batch_id"], session))
    print(f"done ok={sum(results)} failed={len(results) - sum(results)} archive={archive}", flush=True)


if __name__ == "__main__":
    main()
