#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 FILTER" >&2
    echo "  FILTER: I, Y, J, or H" >&2
    echo "  PHASE_THROTTLE controls concurrent SWarp array tasks (default: 4)." >&2
    echo "  Set OVERWRITE=1 to replace existing products." >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

filter=${1^^}
case "$filter" in
    I|Y|J|H) ;;
    *)
        usage
        exit 2
        ;;
esac

phase_throttle=${PHASE_THROTTLE:-4}
if ! [[ "$phase_throttle" =~ ^[1-9][0-9]*$ ]]; then
    echo "PHASE_THROTTLE must be a positive integer." >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
log_dir=/gpfs01/home/ppzhg/logs/ero_psf
mkdir -p "$log_dir"

extra_args=()
if [[ ${OVERWRITE:-0} == 1 ]]; then
    extra_args+=(--overwrite)
fi

printf -v native_command '%q ' \
    "$python" "$script_dir/render_native_epsfs.py" "$filter" "${extra_args[@]}"
native_job=$(sbatch --parsable \
    --partition=defq \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=16g \
    --time=04:00:00 \
    --job-name=epsf_native \
    --output="$log_dir/%j.out" \
    --error="$log_dir/%j.err" \
    --chdir="$script_dir" \
    --wrap="$native_command")
native_job=${native_job%%;*}

printf -v coadd_command '%q ' \
    "$python" "$script_dir/resample_coadd_epsfs.py" "$filter" "${extra_args[@]}"
coadd_job=$(sbatch --parsable \
    --partition=defq \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32g \
    --time=12:00:00 \
    --job-name=epsf_coadd \
    --output="$log_dir/%A_%a.out" \
    --error="$log_dir/%A_%a.err" \
    --chdir="$script_dir" \
    --dependency="afterok:$native_job" \
    --array="0-100%$phase_throttle" \
    --wrap="$coadd_command")
coadd_job=${coadd_job%%;*}

printf -v final_command '%q ' \
    "$python" "$script_dir/build_stack_epsf.py" "$filter" "${extra_args[@]}"
final_job=$(sbatch --parsable \
    --partition=hmemq,defq \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32g \
    --time=12:00:00 \
    --job-name=epsf_stack \
    --output="$log_dir/%j.out" \
    --error="$log_dir/%j.err" \
    --chdir="$script_dir" \
    --dependency="afterok:$coadd_job" \
    --wrap="$final_command")
final_job=${final_job%%;*}

printf -v mge_command '%q ' \
    "$python" "$script_dir/fit_epsf_mge.py" "$filter" "${extra_args[@]}"
mge_job=$(sbatch --parsable \
    --partition=shortq \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=16g \
    --time=01:00:00 \
    --job-name=epsf_mge \
    --output="$log_dir/%j.out" \
    --error="$log_dir/%j.err" \
    --chdir="$script_dir" \
    --dependency="afterok:$final_job" \
    --wrap="$mge_command")
mge_job=${mge_job%%;*}

printf 'Native phase job: %s\n' "$native_job"
printf 'Coadd phase array: %s (0-100, max %s concurrent)\n' \
    "$coadd_job" "$phase_throttle"
printf 'Final stack job:  %s\n' "$final_job"
printf 'MGE fit job:      %s\n' "$mge_job"
