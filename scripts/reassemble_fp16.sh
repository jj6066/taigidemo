#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="$script_dir/models/Breeze-ASR-26-fp16"
output="$model_dir/model-00001-of-00002.safetensors"
temporary="$output.tmp"
part_a="$output.part-00"
part_b="$output.part-01"
expected_bytes=2496411616

if [[ -f "$output" ]] && [[ "$(wc -c < "$output")" -eq "$expected_bytes" ]]; then
    echo "Model shard is already reconstructed: $output"
    exit 0
fi

if [[ ! -f "$part_a" || ! -f "$part_b" ]]; then
    echo "Missing Git LFS model parts. Clone/pull this repository with Git LFS enabled." >&2
    exit 1
fi

cat "$part_a" "$part_b" > "$temporary"
actual_bytes="$(wc -c < "$temporary")"
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    rm -f "$temporary"
    echo "Reassembled shard has $actual_bytes bytes; expected $expected_bytes." >&2
    exit 1
fi

mv "$temporary" "$output"
echo "Reassembled model shard: $output"
