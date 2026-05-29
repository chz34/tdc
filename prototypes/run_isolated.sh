#!/usr/bin/env bash
# Run torchbench:hf_GPT2 across 5 variants, one process per cell for
# crash isolation. Skips the profiling step (long, not useful here).
#
# Each cell includes 'eager' alongside the target variant so the per-
# log speed table has a populated ratio column without needing
# post-processing across files.
#
# Usage:
#     prototypes/run_gpt2_isolated.sh                 # cpu
#     TDC_DEVICE=npu prototypes/run_gpt2_isolated.sh  # npu
#     OUTDIR=mydir prototypes/run_gpt2_isolated.sh    # custom log dir
#
# A cell that crashes (e.g. v3-fallback hitting the upstream proxy
# executor codegen bug on hf_GPT2) only loses its own log -- subsequent
# cells still run.

set -u  # catch typos, but NOT -e: we expect some cells to fail

# Run from torch_dispatch_capture/ regardless of where the user invoked.
cd "$(dirname "$0")/.."

WORKLOAD="hf_GPT2"
VARIANTS=("dynamo" "aot_eager" "inductor" "v2" "v3-fallback")
DEVICE="${TDC_DEVICE:-cpu}"
OUTDIR="${OUTDIR:-prototypes/out/gpt2_isolated_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTDIR"
echo "# v2_benchmark isolated runner"
echo "# device:    $DEVICE"
echo "# workload:  torchbench:$WORKLOAD"
echo "# variants:  ${VARIANTS[*]}"
echo "# outdir:    $OUTDIR"
echo

declare -a summary
for v in "${VARIANTS[@]}"; do
    # Sanitize the variant name into a filename-safe slug. Currently all
    # five names are already safe, but the rule protects future renames.
    slug="${v//[^a-zA-Z0-9_-]/_}"
    log="$OUTDIR/$slug.log"
    echo "==> $v"
    start=$(date +%s)
    TDC_TORCHBENCH=1 TDC_DEVICE="$DEVICE" python prototypes/v2_benchmark.py \
        --variants "$v,eager" \
        --workloads "$WORKLOAD" \
        --skip-profile \
        > "$log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))
    if [ $rc -eq 0 ]; then
        status="PASS"
    else
        status="FAIL (rc=$rc)"
    fi
    summary+=("  $v: $status in ${elapsed}s -> $log")
    echo "    $status (${elapsed}s)"
done

echo
echo "# summary"
printf '%s\n' "${summary[@]}"
