#!/usr/bin/env bash
# Sample memory residency of a running process until it exits.
#
#   ./sample-proc.sh <pid> <out.csv> [interval_seconds]
#
# The columns that matter for PLAN section 1.2:
#   rss_anon  - anonymous (heap) resident memory. This is the swappable kind.
#               If --repack converts mmap'd weights into owned buffers, this is
#               where the ~13 GB shows up.
#   rss_file  - file-backed resident memory, i.e. the mmap'd weights. Clean pages;
#               Linux evicts these, it never swaps them.
#   vm_swap   - actually swapped out. Should stay at ~0 on the mmap path.
#   majflt    - major faults, cumulative. This is the real cost signal once the
#               CPU-side working set exceeds page cache.
#
# Note on host_avail_mb: MemAvailable counts reclaimable page cache as available, and
# the mmap'd weights ARE reclaimable page cache. So it stays high precisely when those
# weights are about to be evicted, and it is NOT a useful pressure signal here. Watch
# majflt instead.
set -euo pipefail

PID="${1:?usage: sample-proc.sh <pid> <out.csv> [interval]}"
OUT="${2:?usage: sample-proc.sh <pid> <out.csv> [interval]}"
INTERVAL="${3:-1}"

echo "t_s,rss_anon_mb,rss_file_mb,rss_shmem_mb,vm_swap_mb,majflt,minflt,gpu_used_mb,host_avail_mb" > "$OUT"

T0="$(date +%s)"
while kill -0 "$PID" 2>/dev/null; do
    ST="/proc/$PID/status"
    STAT="/proc/$PID/stat"
    [ -r "$ST" ] && [ -r "$STAT" ] || break

    anon=$(awk '/^RssAnon:/  {print int($2/1024)}' "$ST" 2>/dev/null || echo "")
    file=$(awk '/^RssFile:/  {print int($2/1024)}' "$ST" 2>/dev/null || echo "")
    shm=$( awk '/^RssShmem:/ {print int($2/1024)}' "$ST" 2>/dev/null || echo "")
    swp=$( awk '/^VmSwap:/   {print int($2/1024)}' "$ST" 2>/dev/null || echo "")

    # Strip "pid (comm) " first - comm can contain spaces and parens. After that
    # the remaining fields start at "state", so minflt is field 8 and majflt 10.
    faults=$(sed 's/.*) //' "$STAT" 2>/dev/null | awk '{print $10","$8}' || echo ",")

    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)

    echo "$(( $(date +%s) - T0 )),${anon},${file},${shm},${swp},${faults},${gpu:-},${avail}" >> "$OUT"
    sleep "$INTERVAL"
done
