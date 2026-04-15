#!/usr/bin/env bash
# launch one Slurm job per (Δz, Δlog M*, q, filter) bin
#
# Redshift bins : 0.2 ≤ z < 1.0 in steps of 0.1
# Mass bins     : (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 12.0)
# q bins        : (0.0, 0.5), (0.5, 1.0), (0.0, 1.0)
# Filters       : I, Y, J, H
# 
# Command run: sbatch ~/Q1_gal_stacks_rot/deconv.py z1 z2 m1 m2 q1 q2 filter

module load miniforge3-uoneasy/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate icl-py313

z_min=0.5   ; z_max=0.8   ; dz=0.1
mass_bins=("9.0 9.5" "9.5 10.0" "10.0 10.5" "10.5 11.0" "11.0 11.5" "11.5 12.0")
q_bins=("0.0 0.5" "0.5 1.0" "0.0 1.0")
filters=("I" "Y" "J" "H")

# Initialize (mass, q) bin index counter (accumulates across all z bins)
mq_bin_index=0

# Loop over z bins
for z1 in $(seq $z_min $dz "$(echo "$z_max - $dz" | bc)"); do
    z2=$(printf "%.1f" "$(echo "$z1 + $dz" | bc)")
    # Loop over mass bins
    for mass_bin in "${mass_bins[@]}"; do
        read -r m1 m2 <<< "$mass_bin"

        # Calculate memory: 17GB at z=0.2, decreasing linearly to 12GB at z=0.7.
        # For z > 0.7, keep the request at the 12GB floor.
        memory=$(printf "%.0f" "$(echo "17 - 10 * ($z1 - 0.2)" | bc)")
        if (( memory < 12 )); then
            memory=12
        fi

        # Loop over axis ratio bins
        for q_bin in "${q_bins[@]}"; do
            read -r q1 q2 <<< "$q_bin"

            # Calculate delay in minutes (2 minutes per mass/q bin)
            delay_minutes=$((mq_bin_index * 2))

            # Loop over filters
            for filter in "${filters[@]}"; do
                echo "Submitting bin z=[${z1},${z2}) logM=[${m1},${m2}) q=[${q1},${q2}) for filter ${filter} with ${memory}GB memory (delay: ${delay_minutes} min)"
                sbatch --begin=now+${delay_minutes}minutes --mem=${memory}g ~/Q1_gal_stacks_rot/deconv.py "$z1" "$z2" "$m1" "$m2" "$q1" "$q2" "$filter"
            done

            # Increment bin index for next (mass, q) bin
            mq_bin_index=$((mq_bin_index + 1))
        done
    done
done
