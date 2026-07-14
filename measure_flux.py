#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=shortq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=5g
#SBATCH --time=1:00:00
#SBATCH --job-name=measure_flux
#SBATCH --output=/gpfs01/home/ppzhg/logs/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/%j.err
# fmt: on

import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

# Ensure sibling modules are importable even when Slurm copies this script to
# `/tmp/slurmd/.../slurm_script` before execution.
PROJECT_DIR = Path("~/Q1_gal_stacks_rot").expanduser().resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from measure_hlr import SBP, build_sbp_path, get_sbps, midpoint, pixel_scale, stack_dir

cosmo = FlatLambdaCDM(
    H0=67.74 * u.km / u.s / u.Mpc,  # Hubble constant
    Om0=0.3089,  # Matter density parameter
    Ob0=0.04860,  # Baryon density parameter
    Tcmb0=2.7255 * u.K,  # CMB temperature
)

ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
HLR_TABLE_PATH = stack_dir / "hlr.txt"
HLR_TABLE = Table.read(HLR_TABLE_PATH, format="ascii.fixed_width")
REF_WAV = {
    "I": 7180.870626325479,
    "Y": 10812.431007172765,
    "J": 13669.531078314576,
    "H": 17707.85463376991,
}
TARGET_REST_WAV = REF_WAV["I"]
RESTFRAME_FILTER_LABEL = "I_rest"


def _prepare_sbp_arrays(sbp):
    """Return cleaned, strictly increasing SBP arrays for integration."""
    sma = np.asarray(sbp.radius, dtype=np.float64)
    intens = np.asarray(sbp.sbp, dtype=np.float64)
    axis_ratio = np.clip(np.asarray(sbp.axis_ratio, dtype=np.float64), 0.0, 1.0)

    valid = np.isfinite(sma) & np.isfinite(intens) & np.isfinite(axis_ratio)
    sma = sma[valid]
    intens = intens[valid]
    axis_ratio = axis_ratio[valid]
    if sma.size == 0:
        return None, None, None

    order = np.argsort(sma)
    sma = sma[order]
    intens = intens[order]
    axis_ratio = axis_ratio[order]

    unique_mask = np.concatenate(([True], np.diff(sma) > 0))
    sma = sma[unique_mask]
    intens = intens[unique_mask]
    axis_ratio = axis_ratio[unique_mask]

    if sma[0] > 0:
        sma = np.insert(sma, 0, 0.0)
        intens = np.insert(intens, 0, intens[0])
        axis_ratio = np.insert(axis_ratio, 0, axis_ratio[0])
    return sma, intens, axis_ratio


def measure_sbp_flux(sbp, radius):
    """
    Measure cumulative flux within a radius by integrating the SBP directly.

    This uses the same elliptical-annulus area approximation as
    `measure_hlr.measure_re_from_sbp`.
    """
    if not np.isfinite(radius):
        return np.nan
    if radius <= 0:
        return 0.0

    sma, intens, axis_ratio = _prepare_sbp_arrays(sbp)
    if sma is None:
        return np.nan

    radius = min(float(radius), float(sma[-1]))
    if radius <= 0:
        return 0.0

    boundaries = np.concatenate(([0.0], sma[(sma > 0) & (sma < radius)], [radius]))
    boundaries = np.unique(boundaries)
    if boundaries.size < 2:
        return 0.0

    intens_at_bounds = np.interp(boundaries, sma, intens)
    q_at_bounds = np.interp(boundaries, sma, axis_ratio)

    annulus_areas = (
        np.pi
        * 0.5
        * (q_at_bounds[:-1] + q_at_bounds[1:])
        * (boundaries[1:] ** 2 - boundaries[:-1] ** 2)
    )
    intens_avg = 0.5 * (intens_at_bounds[:-1] + intens_at_bounds[1:])
    flux_annuli = intens_avg * annulus_areas
    return float(np.nansum(flux_annuli))


def measure_sbp_annulus_fraction(sbp, radius_in, radius_out, norm_radius):
    """Measure annular SBP flux fraction relative to a normalization aperture."""
    if (
        not np.isfinite(radius_in)
        or not np.isfinite(radius_out)
        or not np.isfinite(norm_radius)
        or radius_out <= radius_in
        or norm_radius <= 0
    ):
        return np.nan

    sma, _, _ = _prepare_sbp_arrays(sbp)
    if sma is None:
        return np.nan
    max_radius = float(sma[-1])
    if radius_in >= max_radius or radius_out > max_radius or norm_radius > max_radius:
        return np.nan

    flux_norm = measure_sbp_flux(sbp, norm_radius)
    if flux_norm == 0 or not np.isfinite(flux_norm):
        return np.nan

    flux_out = measure_sbp_flux(sbp, radius_out)
    flux_in = measure_sbp_flux(sbp, radius_in)
    return (flux_out - flux_in) / flux_norm


def lookup_re_sbp_deconv(avg_z, avg_mstar, avg_q, filter_name, gal_type):
    """Fetch the imcascade SBP half-light radius from hlr.txt."""
    mask = (
        np.isclose(np.asarray(HLR_TABLE["z"], dtype=float), avg_z)
        & np.isclose(np.asarray(HLR_TABLE["mstar"], dtype=float), avg_mstar)
        & np.isclose(np.asarray(HLR_TABLE["q"], dtype=float), avg_q)
        & (np.asarray(HLR_TABLE["filter"]) == filter_name)
        & (np.asarray(HLR_TABLE["gal_type"]) == gal_type)
    )
    matches = np.where(mask)[0]
    if matches.size != 1:
        return np.nan
    return float(HLR_TABLE["re_sbp_deconv"][matches[0]])


def load_bootstrap_sbps(z1, z2, m1, m2, q1, q2, filter_name, gal_type):
    """Load background-subtracted bootstrap SBPs from the ASDF products."""
    asdf_path = build_sbp_path(
        z1,
        z2,
        m1,
        m2,
        q1,
        q2,
        filter_name,
        gal_type,
        use_reference_isophotes=True,
    )
    if not asdf_path.exists():
        return []

    with asdf.open(asdf_path) as sbps_dict:
        pixel_to_kpc = float(sbps_dict.tree["pixel_to_kpc"])
        sbp_bs_tables = sbps_dict.tree["bootstrap"]
        res = []
        for tab in sbp_bs_tables:
            sma = np.asarray(tab["sma"], dtype=np.float64)
            intens = np.asarray(tab["intens"], dtype=np.float64)
            axis_ratio = np.clip(
                1.0 - np.asarray(tab["ellipticity"], dtype=np.float64),
                0.0,
                1.0,
            )
            res.append(
                SBP(
                    radius=sma,
                    radius_kpc=sma * pixel_to_kpc,
                    sbp=intens,
                    sbp_err=np.full_like(intens, np.nan),
                    axis_ratio=axis_ratio,
                )
            )
    return res


def bootstrap_fractional_error(values):
    """Return bootstrap fractional scatter, mirroring the old pipeline."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    mean_val = float(np.nanmean(values))
    if not np.isfinite(mean_val) or np.isclose(mean_val, 0.0):
        return np.nan
    return float(np.nanstd(values) / np.abs(mean_val))


def scale_bootstrap_error(nominal_value, bootstrap_values):
    """Scale the nominal measurement by the bootstrap fractional scatter."""
    frac_err = bootstrap_fractional_error(bootstrap_values)
    if not np.isfinite(nominal_value) or not np.isfinite(frac_err):
        return np.nan, np.nan
    return float(np.abs(nominal_value) * frac_err), frac_err


def interpolate_restframe_i_value(group_rows, value_key):
    """
    Interpolate one value column to rest-frame I.

    Use only the nearest bracketing filters in log(rest_wavelength) space to
    reduce sensitivity to outliers in the more distant bands.
    """
    rows = []
    for row in group_rows:
        filt = row["filter"]
        if filt not in REF_WAV:
            continue
        value = row[value_key]
        rest_wav = REF_WAV[filt] / (1.0 + row["z"])
        rows.append((rest_wav, value))
    if len(rows) == 0:
        return np.nan

    rows.sort(key=lambda item: item[0])
    rest_wav = np.array([item[0] for item in rows], dtype=np.float64)
    values = np.array([item[1] for item in rows], dtype=np.float64)

    exact = np.isclose(rest_wav, TARGET_REST_WAV, rtol=0.0, atol=1e-8)
    if np.any(exact):
        value = values[np.where(exact)[0][0]]
        return float(value) if np.isfinite(value) else np.nan
    if TARGET_REST_WAV < rest_wav[0] or TARGET_REST_WAV > rest_wav[-1]:
        return np.nan

    hi = int(np.searchsorted(rest_wav, TARGET_REST_WAV, side="right"))
    lo = hi - 1
    if lo < 0 or hi >= rest_wav.size:
        return np.nan
    if not np.isfinite(values[lo]) or not np.isfinite(values[hi]):
        return np.nan

    x = np.log10(rest_wav[[lo, hi]])
    y = values[[lo, hi]]
    xt = np.log10(TARGET_REST_WAV)
    return float(np.interp(xt, x, y))


def append_restframe_i_rows(results):
    """Append synthetic rest-frame-I rows built from the filter measurements."""
    grouped = defaultdict(list)
    for row in results:
        key = (row["z"], row["mstar"], row["q"], row["gal_type"])
        grouped[key].append(row)

    ratio_keys = (
        "flux_ratio_3re_6re",
        "flux_ratio_5re_9re",
        "flux_ratio_6re_10re",
        "flux_ratio_gt10kpc",
    )
    frac_err_keys = tuple(f"{key}_frac_err" for key in ratio_keys)
    appended = []
    for (z, mstar, q, gal_type), group_rows in sorted(grouped.items()):
        interp_values = {
            key: interpolate_restframe_i_value(group_rows, key) for key in ratio_keys
        }
        interp_frac_errs = {
            key: interpolate_restframe_i_value(group_rows, key) for key in frac_err_keys
        }
        interp_errors = {
            f"{key}_err": (
                np.abs(interp_values[key]) * interp_frac_errs[f"{key}_frac_err"]
                if np.isfinite(interp_values[key])
                and np.isfinite(interp_frac_errs[f"{key}_frac_err"])
                else np.nan
            )
            for key in ratio_keys
        }
        if not any(np.isfinite(val) for val in interp_values.values()):
            continue
        pixel_to_kpc = float(group_rows[0]["pixel_to_kpc"])
        appended.append(
            {
                "z": z,
                "mstar": mstar,
                "q": q,
                "gal_type": gal_type,
                "filter": RESTFRAME_FILTER_LABEL,
                "pixel_to_kpc": pixel_to_kpc,
                "re_sbp_deconv": np.nan,
                "r_3sig": np.nan,
                "flux_3sig": np.nan,
                "flux_3sig_err": np.nan,
                **interp_values,
                **interp_errors,
            }
        )
    return results + appended


def process_single_measurement(args):
    """Worker function to process a single (z, m, q, filter) combination."""
    z1, z2, m1, m2, q1, q2, filter_name, gal_type = args

    avg_z = midpoint(z1, z2)
    avg_mstar = midpoint(m1, m2)
    avg_q = midpoint(q1, q2)
    angular_diameter_distance = cosmo.angular_diameter_distance(avg_z)
    pixel_scale_rad = (pixel_scale * u.arcsec).to(u.rad)
    pixel_to_mpc = pixel_scale_rad.value * angular_diameter_distance
    pixel_to_kpc = pixel_to_mpc.to(u.kpc).value

    re_sbp_deconv = lookup_re_sbp_deconv(
        avg_z, avg_mstar, avg_q, filter_name, gal_type
    )
    if not np.isfinite(re_sbp_deconv) or re_sbp_deconv <= 0:
        print(
            "Missing valid re_sbp_deconv in hlr.txt for"
            f" z:{z1}-{z2}, m:{m1}-{m2}, q:{q1}-{q2},"
            f" {filter_name}-band, gal_type={gal_type}"
        )
        return None

    _, imcascade_sbp, _ = get_sbps(
        z1,
        z2,
        m1,
        m2,
        q1,
        q2,
        filter_name,
        gal_type,
        use_reference_isophotes=True,
    )
    if imcascade_sbp is None or imcascade_sbp.radius.size < 2:
        print(
            "Insufficient SBP data for"
            f" z:{z1}-{z2}, m:{m1}-{m2}, q:{q1}-{q2},"
            f" {filter_name}-band, gal_type={gal_type}"
        )
        return None

    rad_3sig = float(imcascade_sbp.radius[-1])
    rad_3re = 3.0 * re_sbp_deconv
    rad_5re = 5.0 * re_sbp_deconv
    rad_6re = 6.0 * re_sbp_deconv
    rad_9re = 9.0 * re_sbp_deconv
    rad_10re = 10.0 * re_sbp_deconv
    rad_10kpc = 10.0 / pixel_to_kpc

    flux_3sig = measure_sbp_flux(imcascade_sbp, rad_3sig)
    flux_ratio_3re_6re = measure_sbp_annulus_fraction(
        imcascade_sbp, rad_3re, rad_6re, rad_3sig
    )
    flux_ratio_5re_9re = measure_sbp_annulus_fraction(
        imcascade_sbp, rad_5re, rad_9re, rad_3sig
    )
    flux_ratio_6re_10re = measure_sbp_annulus_fraction(
        imcascade_sbp, rad_6re, rad_10re, rad_3sig
    )
    flux_ratio_gt10kpc = measure_sbp_annulus_fraction(
        imcascade_sbp, rad_10kpc, rad_3sig, rad_3sig
    )

    bs_sbps = load_bootstrap_sbps(z1, z2, m1, m2, q1, q2, filter_name, gal_type)
    flux_3sig_bs = [measure_sbp_flux(bs_sbp, rad_3sig) for bs_sbp in bs_sbps]
    flux_ratio_3re_6re_bs = [
        measure_sbp_annulus_fraction(bs_sbp, rad_3re, rad_6re, rad_3sig)
        for bs_sbp in bs_sbps
    ]
    flux_ratio_5re_9re_bs = [
        measure_sbp_annulus_fraction(bs_sbp, rad_5re, rad_9re, rad_3sig)
        for bs_sbp in bs_sbps
    ]
    flux_ratio_6re_10re_bs = [
        measure_sbp_annulus_fraction(bs_sbp, rad_6re, rad_10re, rad_3sig)
        for bs_sbp in bs_sbps
    ]
    flux_ratio_gt10kpc_bs = [
        measure_sbp_annulus_fraction(bs_sbp, rad_10kpc, rad_3sig, rad_3sig)
        for bs_sbp in bs_sbps
    ]

    flux_3sig_err, flux_3sig_frac_err = scale_bootstrap_error(
        flux_3sig, flux_3sig_bs
    )
    flux_ratio_3re_6re_err, flux_ratio_3re_6re_frac_err = scale_bootstrap_error(
        flux_ratio_3re_6re, flux_ratio_3re_6re_bs
    )
    flux_ratio_5re_9re_err, flux_ratio_5re_9re_frac_err = scale_bootstrap_error(
        flux_ratio_5re_9re, flux_ratio_5re_9re_bs
    )
    flux_ratio_6re_10re_err, flux_ratio_6re_10re_frac_err = scale_bootstrap_error(
        flux_ratio_6re_10re, flux_ratio_6re_10re_bs
    )
    flux_ratio_gt10kpc_err, flux_ratio_gt10kpc_frac_err = scale_bootstrap_error(
        flux_ratio_gt10kpc, flux_ratio_gt10kpc_bs
    )

    print(
        f"z:{z1}-{z2}, m:{m1}-{m2}, q:{q1}-{q2}, {filter_name}-band "
        f"gal_type={gal_type}, "
        f"re_sbp_deconv={re_sbp_deconv * pixel_to_kpc:.2f} kpc, "
        f"flux_3sig={flux_3sig:.3e}, "
        f"f(3re,6re)={flux_ratio_3re_6re:.4f}±{flux_ratio_3re_6re_err:.4f}, "
        f"f(5re,9re)={flux_ratio_5re_9re:.4f}±{flux_ratio_5re_9re_err:.4f}, "
        f"f(6re,10re)={flux_ratio_6re_10re:.4f}±{flux_ratio_6re_10re_err:.4f}, "
        f"f(>10kpc)={flux_ratio_gt10kpc:.4f}±{flux_ratio_gt10kpc_err:.4f}"
    )

    return {
        "z": avg_z,
        "mstar": avg_mstar,
        "q": avg_q,
        "gal_type": gal_type,
        "filter": filter_name,
        "pixel_to_kpc": pixel_to_kpc,
        "re_sbp_deconv": re_sbp_deconv,
        "r_3sig": rad_3sig,
        "flux_3sig": flux_3sig,
        "flux_3sig_err": flux_3sig_err,
        "flux_ratio_3re_6re": flux_ratio_3re_6re,
        "flux_ratio_3re_6re_err": flux_ratio_3re_6re_err,
        "flux_ratio_5re_9re": flux_ratio_5re_9re,
        "flux_ratio_5re_9re_err": flux_ratio_5re_9re_err,
        "flux_ratio_6re_10re": flux_ratio_6re_10re,
        "flux_ratio_6re_10re_err": flux_ratio_6re_10re_err,
        "flux_ratio_gt10kpc": flux_ratio_gt10kpc,
        "flux_ratio_gt10kpc_err": flux_ratio_gt10kpc_err,
        "flux_ratio_3re_6re_frac_err": flux_ratio_3re_6re_frac_err,
        "flux_ratio_5re_9re_frac_err": flux_ratio_5re_9re_frac_err,
        "flux_ratio_6re_10re_frac_err": flux_ratio_6re_10re_frac_err,
        "flux_ratio_gt10kpc_frac_err": flux_ratio_gt10kpc_frac_err,
    }


if __name__ == "__main__":
    if os.environ.get("MEASURE_FLUX_SMOKE_TEST", "").strip() in {"1", "true", "TRUE"}:
        print(f"measure_flux smoke test import succeeded from {PROJECT_DIR}")
        raise SystemExit(0)

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
        (0.8, 0.9),
        (0.9, 1.0),
    ]
    q_bins = [(0.0, 0.5), (0.5, 1.0), (0.0, 1.0)]
    filters = ["I", "Y", "J", "H"]
    gal_types = ["lcg", "hcg"]

    task_args = [
        (z1, z2, m1, m2, q1, q2, filter_name, gal_type)
        for z1, z2 in z_bins
        for m1, m2 in m_bins
        for q1, q2 in q_bins
        for filter_name in filters
        for gal_type in gal_types
    ]

    print(
        f"Processing {len(task_args)} measurements using {min(len(task_args), ncores)} cores..."
    )
    with mp.Pool(processes=min(len(task_args), ncores)) as pool:
        results = pool.map(process_single_measurement, task_args)

    results = [r for r in results if r is not None]
    results = append_restframe_i_rows(results)

    result_table = Table(
        data={
            "z": [r["z"] for r in results],
            "mstar": [r["mstar"] for r in results],
            "q": [r["q"] for r in results],
            "gal_type": [r["gal_type"] for r in results],
            "filter": [r["filter"] for r in results],
            "pixel_to_kpc": [r["pixel_to_kpc"] for r in results],
            "re_sbp_deconv": [r["re_sbp_deconv"] for r in results],
            "r_3sig": [r["r_3sig"] for r in results],
            "flux_3sig": [r["flux_3sig"] for r in results],
            "flux_3sig_err": [r["flux_3sig_err"] for r in results],
            "flux_ratio_3re_6re": [r["flux_ratio_3re_6re"] for r in results],
            "flux_ratio_3re_6re_err": [r["flux_ratio_3re_6re_err"] for r in results],
            "flux_ratio_5re_9re": [r["flux_ratio_5re_9re"] for r in results],
            "flux_ratio_5re_9re_err": [r["flux_ratio_5re_9re_err"] for r in results],
            "flux_ratio_6re_10re": [r["flux_ratio_6re_10re"] for r in results],
            "flux_ratio_6re_10re_err": [r["flux_ratio_6re_10re_err"] for r in results],
            "flux_ratio_gt10kpc": [r["flux_ratio_gt10kpc"] for r in results],
            "flux_ratio_gt10kpc_err": [r["flux_ratio_gt10kpc_err"] for r in results],
        }
    )
    result_table["z"].info.format = ".2f"
    result_table["mstar"].info.format = ".2f"
    result_table["q"].info.format = ".2f"
    result_table.write(
        stack_dir / "flux_ratios.txt", format="ascii.fixed_width", overwrite=True
    )
