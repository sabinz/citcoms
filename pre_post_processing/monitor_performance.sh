#!/usr/bin/env bash
#
# monitor_performance.sh - samples CPU, memory, and thread usage for a
# running Create_History.py pre-processing job (the main orchestrator,
# every make_history_for_age.py worker, and every short-lived gmt call)
# at a fixed interval, writing a CSV for later analysis.
#
# Usage:
#   ./monitor_performance.sh [output_csv] [interval_seconds]
#
# Defaults: output_csv=perf_monitor_<timestamp>.csv, interval_seconds=5
#
# Run this in a separate terminal (or backgrounded with '&') alongside
# Create_History.py, in any directory - it finds the relevant processes
# by name, not by location. It auto-stops a few samples after it can no
# longer find any matching processes (i.e. the run finished), so you
# don't have to babysit it - Ctrl-C also works at any time.
#
# Written to be portable across the old bash (3.2) that ships as
# /bin/bash on macOS and zsh (the default login shell) - no
# associative arrays or other bash4+ features.

OUT="${1:-perf_monitor_$(date +%Y%m%d_%H%M%S).csv}"
INTERVAL="${2:-5}"
NCORES=$(sysctl -n hw.ncpu)

echo "monitor_performance.sh: sampling every ${INTERVAL}s -> ${OUT} (machine has ${NCORES} logical cores)"
echo "monitor_performance.sh: stop any time with Ctrl-C; auto-stops once the run appears to have finished"

echo "# ncores=${NCORES} interval_s=${INTERVAL} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT"
echo "timestamp,n_history_procs,n_worker_procs,n_gmt_procs,total_cpu_pct,total_mem_mb,total_threads,sys_cpu_user_pct,sys_cpu_sys_pct,sys_cpu_idle_pct" >> "$OUT"

# convert a top-style MEM value ('1776K','29M','1.2G') to MB
mem_to_mb() {
    local val="$1"
    local num unit
    unit="${val: -1}"
    num="${val%[KMG]}"
    case "$unit" in
        K) awk -v n="$num" 'BEGIN{printf "%.3f", n/1024}' ;;
        M) awk -v n="$num" 'BEGIN{printf "%.3f", n}' ;;
        G) awk -v n="$num" 'BEGIN{printf "%.3f", n*1024}' ;;
        *) echo "0" ;;
    esac
}

# true (0) if $1 appears among the remaining args
pid_in_list() {
    local target="$1"; shift
    local p
    for p in "$@"; do
        [ "$p" = "$target" ] && return 0
    done
    return 1
}

empty_streak=0
seen_any=0

while true; do
    history_pids=($(pgrep -f "[C]reate_History.py"))
    worker_pids=($(pgrep -f "[m]ake_history_for_age.py"))
    gmt_pids=($(pgrep -x gmt))

    all_pids=("${history_pids[@]}" "${worker_pids[@]}" "${gmt_pids[@]}")
    ts=$(date -u +%Y-%m-%dT%H:%M:%S)

    if [ "${#all_pids[@]}" -eq 0 ]; then
        if [ "$seen_any" -eq 1 ]; then
            empty_streak=$((empty_streak+1))
            if [ "$empty_streak" -ge 3 ]; then
                echo "monitor_performance.sh: no matching processes for 3 samples in a row - assuming the run finished, stopping."
                break
            fi
        fi
        echo "${ts},0,0,0,0,0,0,,,," >> "$OUT"
        sleep "$INTERVAL"
        continue
    fi

    seen_any=1
    empty_streak=0

    # build repeated -pid flags for a single top snapshot covering
    # every relevant process at once
    pid_args=()
    for p in "${all_pids[@]}"; do
        pid_args+=(-pid "$p")
    done

    # top's first sample always reports 0% CPU (no prior sample to diff
    # against), so take two one-second-apart samples and keep only the
    # last snapshot
    top_out=$(top -l 2 -s 1 "${pid_args[@]}" -stats pid,command,cpu,mem,th 2>/dev/null \
        | awk '/^Processes:/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}')

    sys_line=$(echo "$top_out" | grep "^CPU usage:")
    sys_user=$(echo "$sys_line" | sed -E 's/.*CPU usage: ([0-9.]+)% user.*/\1/')
    sys_sys=$(echo "$sys_line" | sed -E 's/.*user, ([0-9.]+)% sys.*/\1/')
    sys_idle=$(echo "$sys_line" | sed -E 's/.*sys, ([0-9.]+)% idle.*/\1/')

    # data rows start right after the 'PID ...' header line
    data=$(echo "$top_out" | awk '/^PID/{found=1; next} found')

    total_cpu=0
    total_mem_mb=0
    total_threads=0
    n_history=0
    n_worker=0
    n_gmt=0

    while read -r pid comm cpu mem th; do
        [ -z "$pid" ] && continue
        total_cpu=$(awk -v a="$total_cpu" -v b="$cpu" 'BEGIN{printf "%.3f", a+b}')
        total_mem_mb=$(awk -v a="$total_mem_mb" -v b="$(mem_to_mb "$mem")" 'BEGIN{printf "%.3f", a+b}')
        total_threads=$((total_threads + th))

        # classify by WHICH pgrep pattern found this pid, not by top's
        # command name - both scripts just show up as 'python3' there
        if pid_in_list "$pid" "${history_pids[@]}"; then
            n_history=$((n_history+1))
        elif pid_in_list "$pid" "${worker_pids[@]}"; then
            n_worker=$((n_worker+1))
        elif pid_in_list "$pid" "${gmt_pids[@]}"; then
            n_gmt=$((n_gmt+1))
        fi
    done <<< "$data"

    echo "${ts},${n_history},${n_worker},${n_gmt},${total_cpu},${total_mem_mb},${total_threads},${sys_user},${sys_sys},${sys_idle}" >> "$OUT"

    sleep "$INTERVAL"
done

echo "monitor_performance.sh: done. Log written to ${OUT}"
