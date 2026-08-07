#!/usr/bin/env bash
# Submit sweep jobs: sort prerequisite + filter step (sweep1) once, then per-FIXED_REF transfer steps
# (sweep2+sweep3/4/5).
# Usage: bash submit_sweep.sh [NX=192] [NY=192] [NZ=32] [FIXED_REF=0|1|both] [N_TIME_SKIP=1]
#                            [N_SCALES=30] [N_SCALE_JOBS=1] [EXTRA_DEPEND='']
#   FIXED_REF=both  submits transfer jobs for both 0 and 1 (filter runs only once)
#   N_SCALES        number of log-spaced filter scales (must match sweep1_filter_fields.py's --n-scales)
#   N_SCALE_JOBS    how many concurrent jobs to split each stage's scale loop across (default 1 = the
#                   original single-job-per-stage behavior). With >1, each stage fans out into
#                   N_SCALE_JOBS batch jobs covering disjoint scale-index ranges, followed by one
#                   merge job depending on all of them (PBS multi-parent afterok). This exists because
#                   at large resolution the sequential 30-scale loop can exceed a single job's walltime
#                   cap -- see CLAUDE.md.
#   EXTRA_DEPEND    optional extra PBS job ID every first-stage job should additionally wait on (used by
#                   submit_all_pbs.sh to chain the sweep behind its budgeting stage)
NX=192; NY=192; NZ=32; FIXED_REF=0; N_TIME_SKIP=1; N_SCALES=30; N_SCALE_JOBS=1; EXTRA_DEPEND=""
for arg in "$@"; do case $arg in
  NX=*)           NX="${arg#*=}";;
  NY=*)           NY="${arg#*=}";;
  NZ=*)           NZ="${arg#*=}";;
  FIXED_REF=*)    FIXED_REF="${arg#*=}";;
  N_TIME_SKIP=*)  N_TIME_SKIP="${arg#*=}";;
  N_SCALES=*)     N_SCALES="${arg#*=}";;
  N_SCALE_JOBS=*) N_SCALE_JOBS="${arg#*=}";;
  EXTRA_DEPEND=*) EXTRA_DEPEND="${arg#*=}";;
esac; done
SIM="bci_Nx${NX}_Ny${NY}_Nz${NZ}"

if [ "$N_SCALE_JOBS" -gt "$N_SCALES" ]; then
    echo "ERROR: N_SCALE_JOBS ($N_SCALE_JOBS) cannot exceed N_SCALES ($N_SCALES)" >&2
    exit 1
fi

# $PYTHON (for the cost-weighted split below) comes from hpc_env.sh, same as every .pbs script uses.
# Sourced in a subshell-safe way: it purges/reloads modules, which is fine here since this script only
# submits jobs. If it isn't available, compute_ranges() falls back to the even-by-count split.
if [ "$N_SCALE_JOBS" -gt 1 ] && [ -f ../hpc_env.sh ]; then
    source ../hpc_env.sh > /dev/null 2>&1 || true
fi

# Merge/sort jobs only read already-computed files and write the combined result -- no per-scale compute
# loop -- so they override the .pbs headers' 23:59:00 down to 8 hours. Same "static #PBS directives,
# override on the qsub command line" pattern submit_all_pbs.sh already uses for GPU_FLAGS.
SHORT_WALLTIME=(-l walltime=08:00:00)

# A merge job that fails to submit is the one failure this script can't afford to shrug off: its batch jobs
# still run and still write their per-scale checkpoints, so everything *looks* fine right up until nothing
# assembles them and sweep3/4/5 never run -- and the batch jobs' own exit status says nothing is wrong.
# qsub's own stderr is left to pass through; this just makes a missing job ID fatal instead of a blank line
# in the middle of otherwise-successful output.
require_job() {
    local id=$1 what=$2
    if [ -z "$id" ]; then
        echo "ERROR: qsub returned no job ID for $what -- batch jobs may already be queued, but nothing" >&2
        echo "       will merge their output. Fix the qsub failure above, then submit the merge by hand:" >&2
        echo "       qsub -v NX=$NX,NY=$NY,NZ=$NZ,...,MERGE_ONLY=1 <that stage's .pbs>" >&2
        exit 1
    fi
}

# Prints N_SCALE_JOBS lines of "start end" covering [0, N_SCALES).
#
# Splitting evenly by *count* is unbalanced here: the scales are log-spaced, and per-scale cost grows with
# ℓ (a fixed cost plus a term proportional to the Gaussian kernel's width in grid cells -- an empirical fit,
# see CLAUDE.md; the analytically-obvious "cost ∝ ℓ" model was measured and refuted). At 1024x1024 over the
# default 2km-400km range the largest scale costs ~10x the smallest, so an even-by-count split leaves the
# last job holding several times its share. The SCALE_ADVICE lines this prints also report whether the
# chosen N_SCALE_JOBS is above the threshold where the split can actually reach its floor -- see CLAUDE.md.
# sweep1_filter_fields.py --print-scale-ranges does the cost-weighted partition, reusing its own
# scale_min/scale_max/geomspace logic so this can't drift from the scales actually filtered. Falls back to
# the even-by-count split if that call fails for any reason (no $PYTHON on the submit host, unreadable
# input file, ...) -- a worse split, but still correct and complete.
compute_ranges_even() {
    local n=$1 k=$2 base rem start=0 size end j
    base=$((n / k)); rem=$((n % k))
    for ((j=0; j<k; j++)); do
        size=$base
        [ $j -lt $rem ] && size=$((size + 1))
        end=$((start + size))
        echo "$start $end"
        start=$end
    done
}

# Runs the splitter once and separates its two tagged outputs: SCALE_RANGE lines are the ranges (returned
# on stdout for the caller's read loop), SCALE_ADVICE lines report how balanced the split actually is and
# whether a larger N_SCALE_JOBS would help. The advice goes to stderr so it reaches the terminal instead of
# being consumed as job ranges -- submit time is exactly when it's actionable.
compute_ranges() {
    local n=$1 k=$2 all out
    if [ -n "$PYTHON" ] && all=$("$PYTHON" -u sweep1_filter_fields.py --filename "output/${SIM}.nc" \
                                     --n-time-skip "$N_TIME_SKIP" --n-scales "$n" \
                                     --print-scale-ranges "$k" 2>/dev/null) \
       && out=$(echo "$all" | grep '^SCALE_RANGE ') \
       && [ "$(echo "$out" | wc -l)" -eq "$k" ]; then
        echo "$all" | sed -n 's/^SCALE_ADVICE /  note: /p' >&2
        echo "$out" | awk '{print $2, $3}'
        return
    fi
    echo "  (warning: cost-weighted split unavailable, falling back to even-by-count)" >&2
    compute_ranges_even "$n" "$k"
}

#+++ Filter stage (sweep1)
FILTER_DEPEND_FLAG=()
[ -n "$EXTRA_DEPEND" ] && FILTER_DEPEND_FLAG=(-W "depend=afterok:$EXTRA_DEPEND")

if [ "$N_SCALE_JOBS" -le 1 ]; then
    FILTER_NAME="${SIM}_sweep_filter"
    FILTER_DONE_JOB=$(qsub -N "$FILTER_NAME" \
                           -o "logs/${FILTER_NAME}.log" \
                           -e "logs/${FILTER_NAME}.log" \
                           -v NX=$NX,NY=$NY,NZ=$NZ,N_TIME_SKIP=$N_TIME_SKIP,N_SCALES=$N_SCALES \
                           "${FILTER_DEPEND_FLAG[@]}" \
                           sweep_filter.pbs)
    echo "Submitted filter job ($SIM): $FILTER_DONE_JOB"
else
    # Fresh fan-out: clear any per-scale tmp files left behind by an earlier attempt, so the merge job's
    # completeness/staleness check isn't satisfied by debris from a differently-configured run. Only done
    # on the parallel path -- the sequential path keeps its existing resume-from-what's-there behavior.
    rm -rf "output/${SIM}_filtered_velocities_sweep_tmp"

    FILTER_BATCH_JOBS=()
    while read -r START END; do
        BNAME="${SIM}_sweep_filter_batch_${START}_${END}"
        BJOB=$(qsub -N "$BNAME" \
                    -o "logs/${BNAME}.log" \
                    -e "logs/${BNAME}.log" \
                    -v NX=$NX,NY=$NY,NZ=$NZ,N_TIME_SKIP=$N_TIME_SKIP,N_SCALES=$N_SCALES,SCALE_START_IDX=$START,SCALE_END_IDX=$END \
                    "${FILTER_DEPEND_FLAG[@]}" \
                    sweep_filter.pbs)
        echo "Submitted filter batch [$START,$END) ($SIM): $BJOB"
        FILTER_BATCH_JOBS+=("$BJOB")
    done < <(compute_ranges "$N_SCALES" "$N_SCALE_JOBS")

    FILTER_MERGE_DEPEND=$(IFS=:; echo "afterok:${FILTER_BATCH_JOBS[*]}")
    FMNAME="${SIM}_sweep_filter_merge"
    FILTER_DONE_JOB=$(qsub -N "$FMNAME" \
                           -o "logs/${FMNAME}.log" \
                           -e "logs/${FMNAME}.log" \
                           -v NX=$NX,NY=$NY,NZ=$NZ,N_TIME_SKIP=$N_TIME_SKIP,N_SCALES=$N_SCALES,MERGE_ONLY=1 \
                           "${SHORT_WALLTIME[@]}" \
                           -W "depend=$FILTER_MERGE_DEPEND" \
                           sweep_filter.pbs)
    require_job "$FILTER_DONE_JOB" "the sweep1 filter merge"
    echo "Submitted filter merge (depends on ${FILTER_BATCH_JOBS[*]}): $FILTER_DONE_JOB"
fi
#---

#+++ Sort prerequisite (sweep2 needs <stem>_sorted_density{_fixed_ref}.nc; see sweep_sort.pbs)
# Independent of filtering, so it runs concurrently with the filter stage rather than after it. Skipped
# entirely when the file already exists (e.g. the main non-sweep budgeting pipeline already produced it
# for this simulation).
submit_sort_if_needed() {
    local fr=$1 ref_suffix
    [ "$fr" = "1" ] && ref_suffix="_fixed_ref" || ref_suffix=""
    local out="output/${SIM}_sorted_density${ref_suffix}.nc"
    if [ -f "$out" ]; then
        echo "Sorted density already exists for FIXED_REF=$fr, skipping sort job: $out"
        SORT_JOB=""
        return
    fi
    local name="${SIM}_sweep_sort${ref_suffix}"
    SORT_JOB=$(qsub -N "$name" \
                    -o "logs/${name}.log" \
                    -e "logs/${name}.log" \
                    -v NX=$NX,NY=$NY,NZ=$NZ,FIXED_REF=$fr \
                    "${FILTER_DEPEND_FLAG[@]}" \
                    sweep_sort.pbs)
    echo "Submitted sort job FIXED_REF=$fr: $SORT_JOB"
}
#---

#+++ Transfer stage (sweep2 + sweep3/4/5)
submit_transfer() {
    local fr=$1 ref_suffix
    [ "$fr" = "1" ] && ref_suffix="_fixed_ref" || ref_suffix=""

    submit_sort_if_needed "$fr"
    local sort_job=$SORT_JOB

    # Depends on the filter stage (its single job or its merge job) AND, when one was submitted, the sort
    # job -- a multi-parent afterok spanning two independent upstream stages.
    local depend="afterok:${FILTER_DONE_JOB}"
    [ -n "$sort_job" ] && depend="${depend}:${sort_job}"

    if [ "$N_SCALE_JOBS" -le 1 ]; then
        local name="${SIM}_sweep_transfer${ref_suffix}"
        local job
        job=$(qsub -N "$name" \
                   -o "logs/${name}.log" \
                   -e "logs/${name}.log" \
                   -v NX=$NX,NY=$NY,NZ=$NZ,FIXED_REF=$fr \
                   -W "depend=$depend" \
                   sweep_transfer.pbs)
        echo "Submitted transfer job FIXED_REF=$fr (depends on $depend): $job"
        return
    fi

    # Fresh fan-out: clear stale per-scale checkpoints for this ref variant (see the filter stage's note).
    rm -f output/${SIM}_energy_transfer_sweep_checkpoint_l*${ref_suffix}.nc

    local batch_jobs=()
    while read -r START END; do
        local bname="${SIM}_sweep_transfer${ref_suffix}_batch_${START}_${END}"
        local bjob
        bjob=$(qsub -N "$bname" \
                    -o "logs/${bname}.log" \
                    -e "logs/${bname}.log" \
                    -v NX=$NX,NY=$NY,NZ=$NZ,FIXED_REF=$fr,SCALE_START_IDX=$START,SCALE_END_IDX=$END \
                    -W "depend=$depend" \
                    sweep_transfer.pbs)
        echo "Submitted transfer batch FIXED_REF=$fr [$START,$END) (depends on $depend): $bjob"
        batch_jobs+=("$bjob")
    done < <(compute_ranges "$N_SCALES" "$N_SCALE_JOBS")

    local merge_depend
    merge_depend=$(IFS=:; echo "afterok:${batch_jobs[*]}")
    local mname="${SIM}_sweep_transfer${ref_suffix}_merge"
    local mjob
    mjob=$(qsub -N "$mname" \
                -o "logs/${mname}.log" \
                -e "logs/${mname}.log" \
                -v NX=$NX,NY=$NY,NZ=$NZ,FIXED_REF=$fr,MERGE_ONLY=1 \
                "${SHORT_WALLTIME[@]}" \
                -W "depend=$merge_depend" \
                sweep_transfer.pbs)
    require_job "$mjob" "the sweep2 transfer merge (FIXED_REF=$fr)"
    echo "Submitted transfer merge FIXED_REF=$fr (depends on ${batch_jobs[*]}): $mjob"
}

if [ "$FIXED_REF" = "both" ]; then
    submit_transfer 0
    submit_transfer 1
else
    submit_transfer "$FIXED_REF"
fi
#---
