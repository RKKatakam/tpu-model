#!/usr/bin/env bash
# Run one batch on a fresh Colab session and always stop it afterwards.
# usage: scripts/colab_batch.sh BATCH_JSON SESSION [colab new args, e.g. --tpu v5e1]
# Every remote step has a wall-clock limit and retries, because a dropped
# connection can leave `colab exec` hanging while the VM bills.
set -euo pipefail
batch=$1 session=$2
shift 2
repo=$(cd "$(dirname "$0")/.." && pwd)
batch_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_id"])' "$batch")
out="$repo/data/raw/$batch_id/$session"
mkdir -p "$out"

limit() { perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$@"; }
retry() {
  local tries=$1 secs=$2
  shift 2
  for i in $(seq "$tries"); do
    limit "$secs" "$@" && return 0
    echo "[colab_batch] attempt $i/$tries failed: $*" >&2
    sleep 10
  done
  return 1
}

colab usage >"$out/usage_before.txt" 2>&1
cleanup() {
  limit 120 colab log -s "$session" -o "$out/colab_log.md" >/dev/null 2>&1 || true
  retry 3 120 colab stop -s "$session" 2>&1 | tail -2 || true
  colab usage >"$out/usage_after.txt" 2>&1 || true
  grep -h -E 'balance' "$out/usage_before.txt" "$out/usage_after.txt" || true
}
trap cleanup EXIT
limit 900 colab new -s "$session" "$@" 2>&1 | tee "$out/new.txt"
limit 120 colab status -s "$session" >"$out/status.txt" 2>&1 || true

payload="$(mktemp -d)/payload.tgz"
tar czf "$payload" -C "$repo" capture corpus bench batches requirements.lock requirements.in
unpack="$(mktemp -d)/unpack.py"
echo 'import os, tarfile; os.makedirs("/content/tpum", exist_ok=True); tarfile.open("/content/payload.tgz").extractall("/content/tpum", filter="data"); print("unpacked")' >"$unpack"
retry 3 300 colab upload -s "$session" "$payload" /content/payload.tgz
retry 3 180 colab exec -s "$session" -f "$unpack" --timeout 120

# run_batch.py skips configs that already have a record, so a retry resumes.
retry 3 4000 colab exec -s "$session" -f "$repo/capture/run_batch.py" --timeout 3600 \
  --env "TPUM_BATCH=batches/$(basename "$batch")" --env "TPUM_SESSION=$session" 2>&1 | tee -a "$out/run_batch.log"
retry 3 900 colab download -s "$session" "/content/tpum/results/${batch_id}__${session}.tgz" "$out/results.tgz"
tar xzf "$out/results.tgz" -C "$repo/data/raw"
