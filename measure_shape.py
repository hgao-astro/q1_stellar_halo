#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=shortq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=10g
#SBATCH --time=1:00:00
#SBATCH --job-name=measure_shape
#SBATCH --output=/gpfs01/home/ppzhg/logs/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/%j.err
# fmt: on

import argparse
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import asdf
import numpy as np
from astropy.table import Table

# Slurm executes a spool copy, so add the checkout before importing siblings.
PROJECT_DIR = Path("/gpfs01/home/ppzhg/Q1_gal_stacks_rot")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from measure_hlr import (  # noqa: E402
    REF_WAV,
    RESTFRAME_FILTER_LABEL,
    TARGET_REST_WAV,
    build_sbp_path,
    get_sbps,
    midpoint,
    pixel_scale,
    stack_dir,
)


ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
hlr_table = Table.read(stack_dir / "hlr.txt", format="ascii.fixed_width")
inner_exclusion_pix = (1.5 / 2) / pixel_scale

FILTERS = ("I", "Y", "J", "H")
GAL_TYPES = ("lcg", "hcg")
SHAPE_PRODUCT = "gal deconv imcascade"
ISOPHOTE_RADIUS_GROWTH = 1.1
RANGE_NAMES = (
    "axis_ratio_1re_3re",
    "axis_ratio_3re_6re",
    "axis_ratio_1re_5re",
    "axis_ratio_5re_9re",
    "axis_ratio_5kpc_10kpc",
    "axis_ratio_gt10kpc",
)


def lookup_re_img(avg_z, avg_mstar, avg_q, gal_type, filter_name):
    """Return the imcascade-image Re and physical pixel scale for one stack."""
    mask = (
        np.isclose(np.asarray(hlr_table["z"], dtype=float), avg_z)
        & np.isclose(np.asarray(hlr_table["mstar"], dtype=float), avg_mstar)
        & np.isclose(np.asarray(hlr_table["q"], dtype=float), avg_q)
        & (np.asarray(hlr_table["gal_type"]).astype(str) == gal_type)
        & (np.asarray(hlr_table["filter"]).astype(str) == filter_name)
    )
    matches = np.where(mask)[0]
    if matches.size != 1:
        return np.nan, np.nan
    row = matches[0]
    return float(hlr_table["re_img"][row]), float(hlr_table["pixel_to_kpc"][row])


def load_free_axis_ratio_profile(z1, z2, m1, m2, q1, q2, filter_name, gal_type):
    """Load the local imcascade isophote fit, never the I-reference extraction."""
    path = build_sbp_path(
        z1,
        z2,
        m1,
        m2,
        q1,
        q2,
        filter_name,
        gal_type,
        use_reference_isophotes=False,
    )
    if not path.exists():
        return None

    with asdf.open(path) as af:
        reference_filter = str(af.tree["reference filter"])
        reference_product = str(af.tree["reference image product"])
        if reference_filter != filter_name or reference_product != SHAPE_PRODUCT:
            raise ValueError(
                f"{path.name} is not a local {SHAPE_PRODUCT} isophote fit: "
                f"reference_filter={reference_filter}, "
                f"reference_product={reference_product}"
            )
        table = af.tree[SHAPE_PRODUCT]
        radius = np.asarray(table["sma"], dtype=np.float64)
        ellipticity = np.asarray(table["ellipticity"], dtype=np.float64)
        ellipticity_err = np.asarray(table["ellipticity_err"], dtype=np.float64)
        stop_code = np.asarray(table["stop_code"], dtype=int)

    axis_ratio = 1.0 - ellipticity
    valid_fit = (
        np.isfinite(radius)
        & np.isfinite(axis_ratio)
        & np.isfinite(ellipticity_err)
        & (axis_ratio >= 0.0)
        & (axis_ratio <= 1.0)
        & (ellipticity_err > 0.0)
        & (stop_code == 0)
    )
    return radius, axis_ratio, ellipticity_err, valid_fit


def count_expected_isophotes(radius, r_in, r_out):
    """Count the full geometric isophote grid, extrapolating past the table."""
    positive_radius = np.sort(radius[np.isfinite(radius) & (radius > 0)])
    if (
        positive_radius.size == 0
        or not np.all(np.isfinite([r_in, r_out]))
        or r_in <= 0
        or r_out <= r_in
    ):
        return 0

    anchor = positive_radius[0]
    log_growth = np.log(ISOPHOTE_RADIUS_GROWTH)
    # Bounds follow the profile convention: r_in is exclusive and r_out inclusive.
    first_index = int(np.floor(np.log(r_in / anchor) / log_growth + 1e-10)) + 1
    last_index = int(np.floor(np.log(r_out / anchor) / log_growth + 1e-10))
    return max(0, last_index - first_index + 1)


def weighted_axis_ratio(radius, axis_ratio, axis_ratio_err, valid_fit, r_in, r_out):
    """Measure a partial-range weighted mean with an explicit point-count penalty."""
    requested = np.isfinite(radius) & (radius > r_in) & (radius <= r_out)
    n_requested = count_expected_isophotes(radius, r_in, r_out)
    used = requested & valid_fit
    n_used = int(np.sum(used))
    if n_requested == 0 or n_used == 0:
        return np.nan, np.nan, n_used, np.nan
    if n_used > n_requested:
        raise ValueError("More valid isophotes found than expected on the radius grid.")

    values = axis_ratio[used]
    errors = axis_ratio_err[used]
    weights = 1.0 / errors**2
    sum_weights = float(np.sum(weights))
    mean = float(np.sum(weights * values) / sum_weights)
    measurement_err = float(np.sqrt(1.0 / sum_weights))

    sum_weights_sq = float(np.sum(weights**2))
    n_effective = sum_weights**2 / sum_weights_sq
    variance_denom = sum_weights - sum_weights_sq / sum_weights
    if n_used > 1 and n_effective > 1 and variance_denom > 0:
        weighted_variance = float(
            np.sum(weights * (values - mean) ** 2) / variance_denom
        )
        scatter_err = float(np.sqrt(max(weighted_variance, 0.0) / n_effective))
    else:
        scatter_err = 0.0

    valid_fraction = n_used / n_requested
    base_err = max(measurement_err, scatter_err)
    inflated_err = float(base_err / np.sqrt(valid_fraction))
    return mean, inflated_err, n_used, valid_fraction


def process_single_measurement(args):
    """Measure free-fit imcascade axis ratios for one stack."""
    z1, z2, m1, m2, q1, q2, filter_name, gal_type = args
    avg_z = midpoint(z1, z2)
    avg_mstar = midpoint(m1, m2)
    avg_q = midpoint(q1, q2)
    re_img, pixel_to_kpc = lookup_re_img(avg_z, avg_mstar, avg_q, gal_type, filter_name)
    if not np.isfinite(re_img) or re_img <= 0:
        return None

    profile = load_free_axis_ratio_profile(
        z1, z2, m1, m2, q1, q2, filter_name, gal_type
    )
    if profile is None:
        return None
    radius, axis_ratio, axis_ratio_err, valid_fit = profile
    _, imcascade_sbp, _ = get_sbps(
        z1,
        z2,
        m1,
        m2,
        q1,
        q2,
        filter_name,
        gal_type,
        use_reference_isophotes=False,
    )
    r_3sig = (
        float(imcascade_sbp.radius[-1])
        if imcascade_sbp is not None and imcascade_sbp.radius.size > 0
        else np.nan
    )
    inner_radius = max(re_img, inner_exclusion_pix)
    r_5kpc = max(5.0 / pixel_to_kpc, inner_exclusion_pix)
    r_10kpc = 10.0 / pixel_to_kpc
    ranges = {
        "axis_ratio_1re_3re": (inner_radius, 3 * re_img),
        "axis_ratio_3re_6re": (3 * re_img, 6 * re_img),
        "axis_ratio_1re_5re": (inner_radius, 5 * re_img),
        "axis_ratio_5re_9re": (5 * re_img, 9 * re_img),
        "axis_ratio_5kpc_10kpc": (r_5kpc, r_10kpc),
        "axis_ratio_gt10kpc": (r_10kpc, r_3sig),
    }

    result = {
        "z": avg_z,
        "mstar": avg_mstar,
        "q": avg_q,
        "gal_type": gal_type,
        "filter": filter_name,
        "pixel_to_kpc": pixel_to_kpc,
        "re_img": re_img,
        "r_3sig": r_3sig,
    }
    for name, (r_in, r_out) in ranges.items():
        value, error, n_points, valid_fraction = weighted_axis_ratio(
            radius,
            axis_ratio,
            axis_ratio_err,
            valid_fit,
            r_in,
            r_out,
        )
        result[name] = value
        result[f"{name}_err"] = error
        result[f"{name}_n"] = n_points
        result[f"{name}_valid_fraction"] = valid_fraction
    return result


def _restframe_bracket(group_rows):
    rows = sorted(
        (
            REF_WAV[row["filter"]] / (1.0 + row["z"]),
            row,
        )
        for row in group_rows
        if row["filter"] in REF_WAV
    )
    if not rows:
        return None

    wavelengths = np.asarray([item[0] for item in rows], dtype=np.float64)
    exact = np.isclose(wavelengths, TARGET_REST_WAV, rtol=0.0, atol=1e-8)
    if np.any(exact):
        row = rows[np.where(exact)[0][0]][1]
        return row, row, 0.0
    if TARGET_REST_WAV < wavelengths[0] or TARGET_REST_WAV > wavelengths[-1]:
        return None

    hi = int(np.searchsorted(wavelengths, TARGET_REST_WAV, side="right"))
    lo = hi - 1
    if lo < 0 or hi >= len(rows):
        return None
    x_lo, row_lo = rows[lo]
    x_hi, row_hi = rows[hi]
    fraction = (np.log10(TARGET_REST_WAV) - np.log10(x_lo)) / (
        np.log10(x_hi) - np.log10(x_lo)
    )
    return row_lo, row_hi, float(fraction)


def append_restframe_i_rows(results):
    """Append rest-frame I shapes using nearest bracketing observed filters."""
    grouped = defaultdict(list)
    for row in results:
        key = (row["z"], row["mstar"], row["q"], row["gal_type"])
        grouped[key].append(row)

    appended = []
    for (z, mstar, q, gal_type), group_rows in sorted(grouped.items()):
        bracket = _restframe_bracket(group_rows)
        if bracket is None:
            continue
        row_lo, row_hi, fraction = bracket
        result = {
            "z": z,
            "mstar": mstar,
            "q": q,
            "gal_type": gal_type,
            "filter": RESTFRAME_FILTER_LABEL,
            "pixel_to_kpc": float(row_lo["pixel_to_kpc"]),
            "re_img": (1.0 - fraction) * row_lo["re_img"] + fraction * row_hi["re_img"],
            "r_3sig": (1.0 - fraction) * row_lo["r_3sig"] + fraction * row_hi["r_3sig"],
        }
        any_finite = False
        for name in RANGE_NAMES:
            value_lo = row_lo[name]
            value_hi = row_hi[name]
            error_lo = row_lo[f"{name}_err"]
            error_hi = row_hi[f"{name}_err"]
            if np.all(np.isfinite([value_lo, value_hi, error_lo, error_hi])):
                result[name] = (1.0 - fraction) * value_lo + fraction * value_hi
                result[f"{name}_err"] = np.hypot(
                    (1.0 - fraction) * error_lo,
                    fraction * error_hi,
                )
                result[f"{name}_n"] = min(row_lo[f"{name}_n"], row_hi[f"{name}_n"])
                result[f"{name}_valid_fraction"] = min(
                    row_lo[f"{name}_valid_fraction"],
                    row_hi[f"{name}_valid_fraction"],
                )
                any_finite = True
            else:
                result[name] = np.nan
                result[f"{name}_err"] = np.nan
                result[f"{name}_n"] = 0
                result[f"{name}_valid_fraction"] = np.nan
        if any_finite:
            appended.append(result)
    return results + appended


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure free-isophote imcascade axis-ratio profiles."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=stack_dir / "shapes.txt",
        help="Output fixed-width table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    m_bins = [
        (9.0, 9.5),
        (9.5, 10.0),
        (10.0, 10.5),
        (10.5, 11.0),
        (11.0, 11.5),
        (11.5, 12.0),
    ]
    z_bins = [
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
    ]
    q_bins = [(0.0, 0.5), (0.5, 1.0), (0.0, 1.0)]
    task_args = [
        (z1, z2, m1, m2, q1, q2, filter_name, gal_type)
        for z1, z2 in z_bins
        for m1, m2 in m_bins
        for q1, q2 in q_bins
        for filter_name in FILTERS
        for gal_type in GAL_TYPES
    ]

    print(
        f"Processing {len(task_args)} candidate stacks using "
        f"{min(len(task_args), ncores)} cores."
    )
    with mp.Pool(processes=min(len(task_args), ncores)) as pool:
        results = pool.map(process_single_measurement, task_args)
    results = [result for result in results if result is not None]
    results = append_restframe_i_rows(results)
    if not results:
        raise RuntimeError("No shape measurements were produced.")

    column_names = tuple(results[0])
    result_table = Table(
        rows=[[result[name] for name in column_names] for result in results],
        names=column_names,
    )
    for name in ("z", "mstar", "q"):
        result_table[name].info.format = ".2f"
    result_table.write(args.output, format="ascii.fixed_width", overwrite=True)
    print(f"Wrote {len(result_table)} rows to {args.output}")


if __name__ == "__main__":
    main()
