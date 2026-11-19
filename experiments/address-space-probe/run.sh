#!/bin/sh
# Builds and runs the probe; the vmmap dump goes to vmmap.txt next to it.
set -e
dir="$(cd "$(dirname "$0")" && pwd)"
out="${TMPDIR:-/tmp}/address-space-probe"
cc -O1 "$dir/probe.c" -o "$out"
"$out" | tee "$dir/vmmap.txt" | sed -n '1,8p'
