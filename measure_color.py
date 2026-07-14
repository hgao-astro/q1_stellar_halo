#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=shortq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=20g
#SBATCH --time=1:00:00
#SBATCH --job-name=measure_color
#SBATCH --output=/gpfs01/home/ppzhg/logs/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/%j.err
# fmt: on

import argparse
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

# Slurm executes a spool copy, so add the checkout before importing siblings.
PROJECT_DIR = Path("~/Q1_gal_stacks_rot").expanduser().resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from measure_hlr import (  # noqa: E402
    build_deconv_img_path,
    get_sbps,
    midpoint,
    pixel_scale,
    stack_dir,
)

ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
hlr_table = Table.read(stack_dir / "hlr.txt", format="ascii.fixed_width")
profile_center_exclusion = 0.3 / pixel_scale

MODE_LABELS = {
    "profile": "profile_weighted_mean",
    "integrated": "integrated_flux",
}


@dataclass
class CP:
    radius: np.ndarray
    radius_kpc: np.ndarray
    cp: np.ndarray
    cp_err: np.ndarray


def get_color(res_dc1, res_dc2, nsigma=3):
    """Compute a pointwise color profile on the radii common to two SBPs."""
    if res_dc1 is None or res_dc2 is None:
        return None

    def truncate_sbp(sbp):
        try:
            rad_max_ind = np.where(sbp.sbp < nsigma * sbp.sbp_err)[0][0]
        except IndexError:
            rad_max_ind = len(sbp.radius)
        return (
            np.asarray(sbp.radius[:rad_max_ind], dtype=np.float64),
            np.asarray(sbp.radius_kpc[:rad_max_ind], dtype=np.float64),
            np.asarray(sbp.sbp[:rad_max_ind], dtype=np.float64),
            np.asarray(sbp.sbp_err[:rad_max_ind], dtype=np.float64),
        )

    rad1, rad_kpc1, sbp1, sbp1_err = truncate_sbp(res_dc1)
    rad2, _, sbp2, sbp2_err = truncate_sbp(res_dc2)
    common_rad, ind1, ind2 = np.intersect1d(rad1, rad2, return_indices=True)
    if common_rad.size == 0:
        return None

    sbp1 = sbp1[ind1]
    sbp2 = sbp2[ind2]
    sbp1_err = sbp1_err[ind1]
    sbp2_err = sbp2_err[ind2]
    valid = (
        np.isfinite(sbp1)
        & np.isfinite(sbp2)
        & np.isfinite(sbp1_err)
        & np.isfinite(sbp2_err)
        & (sbp1 > 0)
        & (sbp2 > 0)
        & (sbp1_err >= 0)
        & (sbp2_err >= 0)
    )

    cp = np.full(common_rad.shape, np.nan, dtype=np.float64)
    cp_err = np.full(common_rad.shape, np.nan, dtype=np.float64)
    cp[valid] = -2.5 * np.log10(sbp1[valid] / sbp2[valid])
    ratio_frac_err = np.sqrt(
        (sbp1_err[valid] / sbp1[valid]) ** 2 + (sbp2_err[valid] / sbp2[valid]) ** 2
    )
    cp_err[valid] = 2.5 / np.log(10) * ratio_frac_err
    return CP(
        radius=common_rad,
        radius_kpc=rad_kpc1[ind1],
        cp=cp,
        cp_err=cp_err,
    )


def _profile_range_color(cp, radius_in, radius_out, n_bootstrap=1000, seed=42):
    """Return the inverse-variance weighted mean color in one radial range."""
    if (
        cp is None
        or not np.isfinite(radius_in)
        or not np.isfinite(radius_out)
        or radius_out <= radius_in
        or cp.radius.size == 0
        or radius_out > cp.radius[-1]
    ):
        return np.nan, np.nan

    mask = (cp.radius > radius_in) & (cp.radius <= radius_out)
    valid = (
        np.isfinite(cp.cp[mask]) & np.isfinite(cp.cp_err[mask]) & (cp.cp_err[mask] > 0)
    )
    if not np.any(valid):
        return np.nan, np.nan

    cp_data = cp.cp[mask][valid]
    weights = 1.0 / cp.cp_err[mask][valid] ** 2
    color = float(np.sum(cp_data * weights) / np.sum(weights))

    rng = np.random.default_rng(seed)
    bootstrap_colors = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        boot_idx = rng.choice(len(cp_data), size=len(cp_data), replace=True)
        boot_weights = weights[boot_idx]
        bootstrap_colors[i] = np.sum(cp_data[boot_idx] * boot_weights) / np.sum(
            boot_weights
        )
    color_err = max(
        float(np.nanstd(bootstrap_colors)),
        float(np.sqrt(1.0 / np.sum(weights))),
    )
    return color, color_err


def _annulus_flux_and_error(sbp, radius_in, radius_out):
    """Integrate an elliptical SBP annulus and propagate profile-point errors."""
    radius = np.asarray(sbp.radius, dtype=np.float64)
    intensity = np.asarray(sbp.sbp, dtype=np.float64)
    intensity_err = np.asarray(sbp.sbp_err, dtype=np.float64)
    axis_ratio = np.asarray(sbp.axis_ratio, dtype=np.float64)
    valid = (
        np.isfinite(radius)
        & np.isfinite(intensity)
        & np.isfinite(intensity_err)
        & np.isfinite(axis_ratio)
    )
    radius = radius[valid]
    intensity = intensity[valid]
    intensity_err = intensity_err[valid]
    axis_ratio = np.clip(axis_ratio[valid], 0.0, 1.0)
    if radius.size < 2:
        return np.nan, np.nan

    order = np.argsort(radius)
    radius = radius[order]
    intensity = intensity[order]
    intensity_err = intensity_err[order]
    axis_ratio = axis_ratio[order]
    unique = np.concatenate(([True], np.diff(radius) > 0))
    radius = radius[unique]
    intensity = intensity[unique]
    intensity_err = intensity_err[unique]
    axis_ratio = axis_ratio[unique]

    if (
        not np.isfinite(radius_in)
        or not np.isfinite(radius_out)
        or radius_out <= radius_in
        or radius_in < radius[0]
        or radius_out > radius[-1]
    ):
        return np.nan, np.nan

    boundaries = np.concatenate(
        (
            [radius_in],
            radius[(radius > radius_in) & (radius < radius_out)],
            [radius_out],
        )
    )
    q_boundaries = np.interp(boundaries, radius, axis_ratio)
    annulus_areas = (
        np.pi
        * 0.5
        * (q_boundaries[:-1] + q_boundaries[1:])
        * (boundaries[1:] ** 2 - boundaries[:-1] ** 2)
    )

    # Express the trapezoidal annulus integral as weights on the sampled SBP.
    boundary_weights = np.zeros(boundaries.size, dtype=np.float64)
    boundary_weights[:-1] += 0.5 * annulus_areas
    boundary_weights[1:] += 0.5 * annulus_areas
    profile_weights = np.zeros(radius.size, dtype=np.float64)
    for boundary, weight in zip(boundaries, boundary_weights):
        hi = int(np.searchsorted(radius, boundary, side="left"))
        if hi < radius.size and np.isclose(radius[hi], boundary, rtol=0.0, atol=1e-10):
            profile_weights[hi] += weight
            continue
        if hi == 0 or hi == radius.size:
            return np.nan, np.nan
        lo = hi - 1
        fraction = (boundary - radius[lo]) / (radius[hi] - radius[lo])
        profile_weights[lo] += weight * (1.0 - fraction)
        profile_weights[hi] += weight * fraction

    flux = float(np.sum(profile_weights * intensity))
    flux_err = float(np.sqrt(np.sum((profile_weights * intensity_err) ** 2)))
    return flux, flux_err


def _integrated_range_color(sbp1, sbp2, radius_in, radius_out):
    flux1, flux1_err = _annulus_flux_and_error(sbp1, radius_in, radius_out)
    flux2, flux2_err = _annulus_flux_and_error(sbp2, radius_in, radius_out)
    if not np.isfinite(flux1) or not np.isfinite(flux2) or flux1 <= 0 or flux2 <= 0:
        return np.nan, np.nan

    color = float(-2.5 * np.log10(flux1 / flux2))
    color_err = float(2.5 / np.log(10) * np.hypot(flux1_err / flux1, flux2_err / flux2))
    return color, color_err


def lookup_hlr(avg_z, avg_mstar, avg_q, gal_type, filter_name):
    """Return the current imcascade-image Re and pixel scale for one table row."""
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


def _measure_ranges(mode, sbp1, sbp2, cp, re_img, rad_10kpc, rad_3sig, n_bootstrap):
    ranges = (
        ("color_3sig", 0.0, rad_3sig),
        ("color_lt3re", 0.0, 3 * re_img),
        ("color_3re_6re", 3 * re_img, 6 * re_img),
        ("color_lt5re", 0.0, 5 * re_img),
        ("color_5re_9re", 5 * re_img, 9 * re_img),
        ("color_lt6re", 0.0, 6 * re_img),
        ("color_6re_10re", 6 * re_img, 10 * re_img),
        ("color_lt10kpc", 0.0, rad_10kpc),
        ("color_gt10kpc", rad_10kpc, rad_3sig),
    )
    results = {}
    for seed, (name, radius_in, radius_out) in enumerate(ranges, start=41):
        if mode == "profile":
            profile_radius_in = max(radius_in, profile_center_exclusion)
            color, color_err = _profile_range_color(
                cp,
                profile_radius_in,
                radius_out,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        else:
            color, color_err = _integrated_range_color(
                sbp1, sbp2, radius_in, radius_out
            )
        results[name] = color
        results[f"{name}_err"] = color_err
    return results


def process_single_measurement(args):
    """Process one (z, mass, q, filter pair, galaxy type) combination."""
    z1, z2, m1, m2, q1, q2, filter_pair, gal_type, modes, n_bootstrap = args
    filter1, filter2 = filter_pair
    avg_z = midpoint(z1, z2)
    avg_mstar = midpoint(m1, m2)
    avg_q = midpoint(q1, q2)

    deconv_img_path1 = build_deconv_img_path(z1, z2, m1, m2, q1, q2, filter1, gal_type)
    deconv_img_path2 = build_deconv_img_path(z1, z2, m1, m2, q1, q2, filter2, gal_type)
    if not deconv_img_path1.exists() or not deconv_img_path2.exists():
        return []

    _, imcascade_sbp1, _ = get_sbps(
        z1,
        z2,
        m1,
        m2,
        q1,
        q2,
        filter1,
        gal_type,
        use_reference_isophotes=True,
    )
    _, imcascade_sbp2, _ = get_sbps(
        z1,
        z2,
        m1,
        m2,
        q1,
        q2,
        filter2,
        gal_type,
        use_reference_isophotes=True,
    )
    if (
        imcascade_sbp1 is None
        or imcascade_sbp2 is None
        or imcascade_sbp1.radius.size < 2
        or imcascade_sbp2.radius.size < 2
    ):
        return []

    re_img1, pixel_to_kpc1 = lookup_hlr(avg_z, avg_mstar, avg_q, gal_type, filter1)
    re_img2, pixel_to_kpc2 = lookup_hlr(avg_z, avg_mstar, avg_q, gal_type, filter2)
    if not np.all(np.isfinite([re_img1, re_img2, pixel_to_kpc1, pixel_to_kpc2])):
        return []

    re_img = 0.5 * (re_img1 + re_img2)
    pixel_to_kpc = 0.5 * (pixel_to_kpc1 + pixel_to_kpc2)
    rad_10kpc = 10.0 / pixel_to_kpc
    rad_3sig = min(imcascade_sbp1.radius[-1], imcascade_sbp2.radius[-1])
    cp = get_color(imcascade_sbp1, imcascade_sbp2) if "profile" in modes else None

    results = []
    for mode in modes:
        if mode == "profile" and cp is None:
            continue
        color_values = _measure_ranges(
            mode,
            imcascade_sbp1,
            imcascade_sbp2,
            cp,
            re_img,
            rad_10kpc,
            rad_3sig,
            n_bootstrap,
        )
        results.append(
            {
                "z": avg_z,
                "mstar": avg_mstar,
                "q": avg_q,
                "gal_type": gal_type,
                "filter1": filter1,
                "filter2": filter2,
                "mode": MODE_LABELS[mode],
                "pixel_to_kpc": pixel_to_kpc,
                "re_img": re_img,
                "r_3sig": rad_3sig,
                **color_values,
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure imcascade SBP colors using profile means or integrated fluxes."
    )
    parser.add_argument(
        "--mode",
        choices=("profile", "integrated", "both"),
        default="both",
        help="Color estimator to run (default: both).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Radial bootstrap samples for profile-mode uncertainties.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=stack_dir / "colors.txt",
        help="Output fixed-width table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")

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
    filter_pairs = [("I", "H"), ("Y", "H")]
    gal_types = ["lcg", "hcg"]
    modes = ("profile", "integrated") if args.mode == "both" else (args.mode,)

    task_args = [
        (z1, z2, m1, m2, q1, q2, filter_pair, gal_type, modes, args.n_bootstrap)
        for z1, z2 in z_bins
        for m1, m2 in m_bins
        for q1, q2 in q_bins
        for filter_pair in filter_pairs
        for gal_type in gal_types
    ]
    print(
        f"Processing {len(task_args)} measurements using "
        f"{min(len(task_args), ncores)} cores in {', '.join(modes)} mode."
    )
    with mp.Pool(processes=min(len(task_args), ncores)) as pool:
        result_groups = pool.map(process_single_measurement, task_args)
    results = [result for group in result_groups for result in group]
    if not results:
        raise RuntimeError("No color measurements were produced.")

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
