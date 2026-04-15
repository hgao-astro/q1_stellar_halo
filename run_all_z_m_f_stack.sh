#!/usr/bin/env bash
# launch one Slurm job per (Δz, filter) combination
#
# Redshift bins : 0.1 ≤ z < 1.0  in steps of 0.1
# Filters: I, Y, J, H
# 
# Command run: sbatch ~/Q1_gal_stacks_combined_mask/stack_z.py  z1  z2  filter
#
# Usage: ./run_all_z_m_filter_stack.sh [initial_delay_minutes]
#   initial_delay_minutes: Optional delay (in minutes) before the first job starts (default: 0)

module load miniforge3-uoneasy/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate icl-py313

# Optional parameter: initial delay in minutes (default is 0)
initial_delay=${1:-0}

z_min=0.4   ; z_max=0.8   ; dz=0.1
filters=("I" "Y" "J" "H")

# Loop over z bins
z_bin_count=0
for z1 in $(seq $z_min $dz "$(echo "$z_max - $dz" | bc)"); do
    z2=$(printf "%.1f" "$(echo "$z1 + $dz" | bc)")
    # Calculate delay in minutes for this z bin (40 minutes per z bin approx time take to subtract bkg from ~300 tiles to avoid disk IO overload)
    delay_minutes=$((initial_delay + z_bin_count * 60))
    
    # Loop over filters
    filter_count=0
    for filter in "${filters[@]}"; do
        # Add 2-minute buffer between filters within the same z bin
        total_delay_minutes=$((delay_minutes + filter_count * 2))
        
        if [ $total_delay_minutes -eq 0 ]; then
            echo "Submitting bin z=[${z1},${z2}) for filter ${filter} (immediate)"
            sbatch ~/Q1_gal_stacks_rot/stack_z.py "$z1" "$z2" "$filter"
        else
            echo "Submitting bin z=[${z1},${z2}) for filter ${filter} (delayed by ${total_delay_minutes} minutes)"
            sbatch --begin=now+${total_delay_minutes}minutes ~/Q1_gal_stacks_rot/stack_z.py "$z1" "$z2" "$filter"
        fi
        
        filter_count=$((filter_count + 1))
    done
    
    z_bin_count=$((z_bin_count + 1))
done
