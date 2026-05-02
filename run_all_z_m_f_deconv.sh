#!/usr/bin/env bash
# launch one Slurm job per (Δz, Δlog M*, q, filter) bin
#
# Redshift bins : 0.2 ≤ z < 0.8 in steps of 0.1
# Mass bins     : (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 12.0)
# q bins        : (0.0, 0.5), (0.5, 1.0), (0.0, 1.0)
# Filters       : I, Y, J, H
#
# Command run:
#   sbatch ~/Q1_gal_stacks_rot/deconv.py z1 z2 m1 m2 q1 q2 filter --deconv-method METHOD
#
# Usage: ./run_all_z_m_f_deconv.sh DECONV_METHOD
#   DECONV_METHOD: imcascade | wiener | richardson_lucy | pysersic

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 DECONV_METHOD" >&2
    exit 1
fi

deconv_method="$1"
case "$deconv_method" in
    imcascade|wiener|richardson_lucy|pysersic) ;;
    *)
        echo "Invalid deconvolution method: $deconv_method" >&2
        echo "Allowed values: imcascade, wiener, richardson_lucy, pysersic" >&2
        exit 1
        ;;
esac

module load miniforge3-uoneasy/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate icl-py313

deconv_script="$HOME/Q1_gal_stacks_rot/deconv.py"
z_min=0.2   ; z_max=0.8   ; dz=0.1
mass_bins=("9.0 9.5" "9.5 10.0" "10.0 10.5" "10.5 11.0" "11.0 11.5" "11.5 12.0")
q_bins=("0.0 0.5" "0.5 1.0" "0.0 1.0")
filters=("I" "Y" "J" "H")

memory_for_job() {
    local method="$1"
    local z1="$2"

    case "$method" in
        imcascade)
            # Measured from imcascade jobs 6674034-6674465 using MaxRSS, with
            # roughly >=20% headroom: max RSS by z was
            # 10.15, 6.83, 5.44, 4.69, 6.08, 3.90 GB.
            case "$z1" in
                0.2) echo 13 ;;
                0.3) echo 9 ;;
                0.4) echo 8 ;;
                0.5) echo 8 ;;
                0.6) echo 8 ;;
                *) echo 6 ;;
            esac
            ;;
        pysersic)
            # pysersic fits on the central crop and only renders full-frame
            # products after optimization, so it can use a lighter z-dependent
            # table than imcascade.
            case "$z1" in
                0.2) echo 10 ;;
                0.3|0.4) echo 8 ;;
                *) echo 6 ;;
            esac
            ;;
        *)
            # Wiener/RL operate directly on the image/PSF arrays and do not need
            # the large imcascade Gaussian render stack.
            echo 5
            ;;
    esac
}

# Initialize bin index counter for staggered starts across all submissions.
bin_index=0

# Loop over z bins
for z1 in $(seq $z_min $dz "$(echo "$z_max - $dz" | bc)"); do
    z2=$(printf "%.1f" "$(echo "$z1 + $dz" | bc)")
    # Loop over mass bins
    for mass_bin in "${mass_bins[@]}"; do
        read -r m1 m2 <<< "$mass_bin"
        memory=$(memory_for_job "$deconv_method" "$z1")

        # Loop over axis ratio bins
        for q_bin in "${q_bins[@]}"; do
            read -r q1 q2 <<< "$q_bin"
            delay_seconds=$((bin_index * 20))

            # Loop over filters
            for filter in "${filters[@]}"; do
                sbatch_args=(
                    --begin=now+${delay_seconds}seconds
                    --mem=${memory}g
                )
                if [[ "$deconv_method" == "pysersic" ]]; then
                    sbatch_args+=(--cpus-per-task=5)
                fi

                echo \
                    "Submitting ${deconv_method} deconvolution for" \
                    "z=[${z1},${z2}) logM=[${m1},${m2}) q=[${q1},${q2})" \
                    "filter=${filter} mem=${memory}GB" \
                    "cpus=$([[ "$deconv_method" == "pysersic" ]] && echo 5 || echo 1)" \
                    "(delay: ${delay_seconds} s)"
                sbatch "${sbatch_args[@]}" "$deconv_script" \
                    "$z1" "$z2" "$m1" "$m2" "$q1" "$q2" "$filter" \
                    --deconv-method "$deconv_method"
            done
            bin_index=$((bin_index + 1))
        done
    done
done
