#!/usr/bin/env bash
# launch one Slurm job per (z, mass, q) bin
#
# Redshift bins : 0.2 <= z < 1.0 in steps of 0.1
# Mass bins     : (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 12.0)
# q bins        : (0.0, 0.5), (0.5, 1.0), (0.0, 1.0)
#
# Command run: sbatch ~/Q1_gal_stacks_rot/extract_sbps.py z1 z2 m1 m2 q1 q2

module load miniforge3-uoneasy/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate icl-py313

z_min=0.2   ; z_max=0.8   ; dz=0.1
mass_bins=("9.0 9.5" "9.5 10.0" "10.0 10.5" "10.5 11.0" "11.0 11.5" "11.5 12.0")
q_bins=("0.0 0.5" "0.5 1.0" "0.0 1.0")

# Initialize (mass, q) bin index counter (accumulates across all z bins)
mq_bin_index=0

# Loop over z bins
for z1 in $(seq "$z_min" "$dz" "$(echo "$z_max - $dz" | bc)"); do
    z2=$(printf "%.1f" "$(echo "$z1 + $dz" | bc)")

    # Loop over mass bins
    for mass_bin in "${mass_bins[@]}"; do
        read -r m1 m2 <<< "$mass_bin"

        # Loop over axis-ratio bins
        for q_bin in "${q_bins[@]}"; do
            read -r q1 q2 <<< "$q_bin"

            # Calculate delay in minutes (1 minutes per mass/q bin)
            delay_minutes=$((mq_bin_index * 1))

            echo "Submitting SBP extraction for z=[${z1},${z2}) logM=[${m1},${m2}) q=[${q1},${q2}) (delay: ${delay_minutes} min)"
            sbatch --begin=now+${delay_minutes}minutes ~/Q1_gal_stacks_rot/extract_sbps.py \
                "$z1" "$z2" "$m1" "$m2" "$q1" "$q2"

            # Increment bin index for next (mass, q) bin
            mq_bin_index=$((mq_bin_index + 1))
        done
    done
done
