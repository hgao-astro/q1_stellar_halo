#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=shortq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=5g
#SBATCH --time=1:00:00
#SBATCH --job-name=measure_re
#SBATCH --output=/gpfs01/home/ppzhg/logs/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/%j.err
# fmt: on

import multiprocessing as mp
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from photutils.profiles import CurveOfGrowth
from scipy.interpolate import PchipInterpolator

cosmo = FlatLambdaCDM(
    H0=67.74 * u.km / u.s / u.Mpc,  # Hubble constant
    Om0=0.3089,  # Matter density parameter
    Ob0=0.04860,  # Baryon density parameter
    Tcmb0=2.7255 * u.K,  # CMB temperature
)

stack_dir = Path("~/Q1_gal_stacks_rot").expanduser()
pixel_scale = 0.3
ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
REFERENCE_FILTER = "I"
REFERENCE_IMAGE_KIND = "imcascade"
REF_WAV = {
    "I": 7180.870626325479,
    "Y": 10812.431007172765,
    "J": 13669.531078314576,
    "H": 17707.85463376991,
}
TARGET_REST_WAV = REF_WAV["I"]
RESTFRAME_FILTER_LABEL = "I_rest"


@dataclass
class SBP:
    """Container for observed surface brightness profile data"""

    radius: np.ndarray
    radius_kpc: np.ndarray
    sbp: np.ndarray
    sbp_err: np.ndarray
    axis_ratio: np.ndarray


def midpoint(lower, upper, ndigits=6):
    """Return a rounded bin center to suppress floating-point display artifacts."""
    return round(0.5 * (float(lower) + float(upper)), ndigits)


def build_sbp_path(
    z1,
    z2,
    m1,
    m2,
    q1,
    q2,
    filter_name,
    gal_type,
    use_reference_isophotes=False,
):
    path = (
        stack_dir / f"sbps_{gal_type}_{filter_name}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}.asdf"
    )
    if use_reference_isophotes and filter_name != REFERENCE_FILTER:
        return path.with_stem(
            path.stem + f"_refiso-{REFERENCE_FILTER}-{REFERENCE_IMAGE_KIND}"
        )
    return path


def build_deconv_img_path(z1, z2, m1, m2, q1, q2, filter_name, gal_type):
    return (
        stack_dir
        / f"stack_{gal_type}_{filter_name}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}_subbkg_deconv_imcascade.fits"
    )


def build_wiener_img_path(z1, z2, m1, m2, q1, q2, filter_name, gal_type):
    return (
        stack_dir
        / f"stack_{gal_type}_{filter_name}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}_subbkg_deconv_wiener.fits"
    )


def get_sbps(
    z1,
    z2,
    m1,
    m2,
    q1,
    q2,
    filter_name,
    gal_type,
    nsigma=3,
    use_reference_isophotes=False,
):
    """
    Compute surface brightness profiles for original data and deconvolved images.
    """
    asdf_path = build_sbp_path(
        z1,
        z2,
        m1,
        m2,
        q1,
        q2,
        filter_name,
        gal_type,
        use_reference_isophotes=use_reference_isophotes,
    )
    if not asdf_path.exists():
        print(f"No existing sbp file found at {asdf_path}.")
        return None, None, None

    with asdf.open(asdf_path) as sbps_dict:
        pixel_to_kpc = float(sbps_dict.tree["pixel_to_kpc"])
        sbp_gal_table = sbps_dict.tree["gal"]
        sbp_sky_tables = sbps_dict.tree["sky"]
        sbp_bs_tables = sbps_dict.tree["bootstrap"]
        sbp_gal_deconv_imcascade_table = sbps_dict.tree["gal deconv imcascade"]
        sbp_gal_deconv_wiener_table = sbps_dict.tree["gal deconv wiener"]

        sma = np.asarray(sbp_gal_table["sma"], dtype=np.float64)
        intens_gal = np.asarray(sbp_gal_table["intens"], dtype=np.float64)
        q_gal = np.clip(
            1.0 - np.asarray(sbp_gal_table["ellipticity"], dtype=np.float64),
            0.0,
            1.0,
        )
        intens_gal_deconv_imcascade = np.asarray(
            sbp_gal_deconv_imcascade_table["intens"], dtype=np.float64
        )
        q_gal_deconv_imcascade = np.clip(
            1.0
            - np.asarray(
                sbp_gal_deconv_imcascade_table["ellipticity"], dtype=np.float64
            ),
            0.0,
            1.0,
        )
        intens_gal_deconv_wiener = np.asarray(
            sbp_gal_deconv_wiener_table["intens"], dtype=np.float64
        )
        q_gal_deconv_wiener = np.clip(
            1.0
            - np.asarray(sbp_gal_deconv_wiener_table["ellipticity"], dtype=np.float64),
            0.0,
            1.0,
        )
        # combine bootstrap error and background error
        bkg_err = (
            np.nanstd([tab["intens"] for tab in sbp_sky_tables], axis=0)
            if len(sbp_sky_tables) > 0
            else np.zeros_like(intens_gal)
        )
        bs_err = (
            np.nanstd([tab["intens"] for tab in sbp_bs_tables], axis=0)
            if len(sbp_bs_tables) > 0
            else np.zeros_like(intens_gal)
        )
        comb_err = np.fmax(bkg_err, bs_err)

    # truncate sbps at >=3*combined_err
    # find the first point dropping below 3*combined_err
    try:
        rad_max_ind = np.where(
            (intens_gal < nsigma * comb_err)
            | (intens_gal_deconv_imcascade < nsigma * comb_err)
            | (intens_gal_deconv_wiener < nsigma * comb_err)
        )[0][0]
    except IndexError:
        print(
            f"All data points are above threshold in {filter_name}-band, {z1}-{z2}, {m1}-{m2}."
        )
        rad_max_ind = len(sma)
    sma = sma[:rad_max_ind]
    sma_kpc = sma * pixel_to_kpc
    comb_err = comb_err[:rad_max_ind]
    intens_gal = intens_gal[:rad_max_ind]
    q_gal = q_gal[:rad_max_ind]
    intens_gal_deconv_imcascade = intens_gal_deconv_imcascade[:rad_max_ind]
    q_gal_deconv_imcascade = q_gal_deconv_imcascade[:rad_max_ind]
    intens_gal_deconv_wiener = intens_gal_deconv_wiener[:rad_max_ind]
    q_gal_deconv_wiener = q_gal_deconv_wiener[:rad_max_ind]

    obs_res = SBP(
        radius=sma,
        radius_kpc=sma_kpc,
        sbp=intens_gal,
        sbp_err=comb_err,
        axis_ratio=q_gal,
    )
    imcascade_res = SBP(
        radius=sma,
        radius_kpc=sma_kpc,
        sbp=intens_gal_deconv_imcascade,
        sbp_err=comb_err,
        axis_ratio=q_gal_deconv_imcascade,
    )
    wiener_res = SBP(
        radius=sma,
        radius_kpc=sma_kpc,
        sbp=intens_gal_deconv_wiener,
        sbp_err=comb_err,
        axis_ratio=q_gal_deconv_wiener,
    )
    return obs_res, imcascade_res, wiener_res


def measure_re_from_img(img, radii=None):
    """Measure the half-light radius using curve of growth."""
    img = np.nan_to_num(np.asarray(img, dtype=np.float64), nan=0.0)
    center = (img.shape[1] / 2 - 0.5, img.shape[0] / 2 - 0.5)
    if radii is None:
        radii = np.geomspace(1, img.shape[0] / 2 - 1, 50)
    radii = np.unique(np.asarray(radii, dtype=np.float64))
    radii = radii[radii > 0]
    if radii.size == 0:
        return np.nan
    cog = CurveOfGrowth(img, center, radii)
    cog.normalize()
    re = cog.calc_radius_at_ee(0.5)
    return float(re) if re is not None else np.nan


def measure_re_from_sbp(sbp: SBP):
    """Measure the half-light radius by integrating the surface brightness profile."""
    # Convert intensity to flux per annulus
    intens = sbp.sbp  # in counts/s/pixel^2
    sma = sbp.radius  # in pixels
    axis_ratio = np.clip(sbp.axis_ratio, 0.0, 1.0)

    # Calculate annulus boundaries
    # Each annulus i spans from sma[i] to sma[i+1]
    # Since sma[0] = 0, first annulus spans from 0 to sma[1]
    if len(sma) < 2:
        return np.nan

    # Inner edges: sma[i]
    r_inner = sma[:-1]
    # Outer edges: sma[i+1]
    r_outer = sma[1:]

    # Approximate each annulus as an ellipse with the average axis ratio
    # between its inner and outer sampled isophotes.
    q_inner = axis_ratio[:-1]
    q_outer = axis_ratio[1:]
    q_annulus = 0.5 * (q_inner + q_outer)
    annulus_areas = np.pi * q_annulus * (r_outer**2 - r_inner**2)  # in pixels^2

    # Average intensity in each annulus = mean of intensity at inner and outer edges
    intens_inner = intens[:-1]
    intens_outer = intens[1:]
    intens_avg = (intens_inner + intens_outer) / 2

    # Flux in each annulus = average intensity * area
    flux_annuli = intens_avg * annulus_areas  # in counts/s

    # Calculate cumulative flux
    cumulative_flux = np.nancumsum(flux_annuli)

    # Total flux is the last element of cumulative flux
    total_flux = cumulative_flux[-1]
    if total_flux <= 0 or np.isnan(total_flux):
        return np.nan

    # Normalize to get enclosed energy fraction
    enclosed_fraction = cumulative_flux / total_flux

    # Remove any NaN values before interpolation
    valid_mask = ~np.isnan(enclosed_fraction)
    if np.sum(valid_mask) < 2:
        return np.nan

    radii_valid = r_outer[valid_mask]
    fraction_valid = enclosed_fraction[valid_mask]

    # Check boundary conditions
    if fraction_valid[-1] < 0.5:
        return np.nan
    if fraction_valid[0] > 0.5:
        return radii_valid[0]

    fraction_valid = np.maximum.accumulate(fraction_valid)
    unique_mask = np.concatenate(([True], np.diff(fraction_valid) > 0))
    radii_valid = radii_valid[unique_mask]
    fraction_valid = fraction_valid[unique_mask]
    if fraction_valid.size < 2:
        return np.nan

    # Use PchipInterpolator to find radius at 50% enclosed energy
    try:
        interp = PchipInterpolator(fraction_valid, radii_valid, extrapolate=False)
        re = float(interp(0.5))
        return re
    except Exception:
        return np.nan


def interpolate_restframe_i_value(group_rows, value_key):
    """
    Interpolate one radius-like quantity to rest-frame I.

    Use only the nearest two filters that bracket the target wavelength in
    log(rest_wavelength) space to reduce sensitivity to outliers in distant
    bands.
    """
    rows = []
    for row in group_rows:
        filt = row["filter"]
        if filt not in REF_WAV:
            continue
        value = row[value_key]
        if not np.isfinite(value):
            continue
        rest_wav = REF_WAV[filt] / (1.0 + row["z"])
        rows.append((rest_wav, value))
    if len(rows) == 0:
        return np.nan

    rows.sort(key=lambda item: item[0])
    rest_wav = np.array([item[0] for item in rows], dtype=np.float64)
    values = np.array([item[1] for item in rows], dtype=np.float64)

    exact = np.isclose(rest_wav, TARGET_REST_WAV, rtol=0.0, atol=1e-8)
    if np.any(exact):
        return float(values[np.where(exact)[0][0]])
    if TARGET_REST_WAV < rest_wav[0] or TARGET_REST_WAV > rest_wav[-1]:
        return np.nan

    hi = int(np.searchsorted(rest_wav, TARGET_REST_WAV, side="right"))
    lo = hi - 1
    if lo < 0 or hi >= rest_wav.size:
        return np.nan

    x = np.log10(rest_wav[[lo, hi]])
    y = values[[lo, hi]]
    xt = np.log10(TARGET_REST_WAV)
    return float(np.interp(xt, x, y))


def append_restframe_i_rows(results):
    """Append synthetic rest-frame-I rows to the HLR results."""
    grouped = defaultdict(list)
    for row in results:
        key = (row["z"], row["mstar"], row["q"], row["gal_type"])
        grouped[key].append(row)

    re_keys = (
        "re_img",
        "re_sbp_obs",
        "re_sbp_deconv",
        "re_img_wiener",
        "re_sbp_wiener",
    )
    appended = []
    for (z, mstar, q, gal_type), group_rows in sorted(grouped.items()):
        interp_values = {
            key: interpolate_restframe_i_value(group_rows, key) for key in re_keys
        }
        if not any(np.isfinite(val) for val in interp_values.values()):
            continue
        appended.append(
            {
                "z": z,
                "mstar": mstar,
                "q": q,
                "gal_type": gal_type,
                "pixel_to_kpc": float(group_rows[0]["pixel_to_kpc"]),
                "filter": RESTFRAME_FILTER_LABEL,
                **interp_values,
            }
        )
    return results + appended


def process_single_measurement(args):
    """Worker function to process a single (z, m, filter) combination."""
    z1, z2, m1, m2, q1, q2, filter_name, gal_type = args

    avg_z = midpoint(z1, z2)
    angular_diameter_distance = cosmo.angular_diameter_distance(avg_z)
    pixel_scale_rad = (pixel_scale * u.arcsec).to(u.rad)
    pixel_to_mpc = pixel_scale_rad.value * angular_diameter_distance
    pixel_to_kpc = pixel_to_mpc.to(u.kpc).value

    deconv_img_path = build_deconv_img_path(
        z1, z2, m1, m2, q1, q2, filter_name, gal_type
    )
    wiener_img_path = build_wiener_img_path(
        z1, z2, m1, m2, q1, q2, filter_name, gal_type
    )
    if not deconv_img_path.exists() or not wiener_img_path.exists():
        print(
            "Deconvolved image not found:"
            f" imcascade={deconv_img_path.exists()} ({deconv_img_path}),"
            f" wiener={wiener_img_path.exists()} ({wiener_img_path})"
        )
        return None

    obs_sbp, imcascade_sbp, wiener_sbp = get_sbps(
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
    if (
        obs_sbp is None
        or imcascade_sbp is None
        or wiener_sbp is None
        or obs_sbp.radius.size < 5
    ):
        print(
            f"Insufficient data points in sbp at z: {z1}-{z2}, m: {m1}-{m2}, q: {q1}-{q2}, {filter_name} band"
        )
        return None

    # re from integrating sbp
    re_sbp_obs = measure_re_from_sbp(obs_sbp)
    re_sbp_deconv = measure_re_from_sbp(imcascade_sbp)
    re_sbp_wiener = measure_re_from_sbp(wiener_sbp)

    # re from curve of growth on the imcascade deconvolved image
    img = fits.getdata(deconv_img_path)
    re_img = measure_re_from_img(img, radii=imcascade_sbp.radius[1:])
    wiener_img = fits.getdata(wiener_img_path)
    re_img_wiener = measure_re_from_img(wiener_img, radii=wiener_sbp.radius[1:])

    print(
        f"z:{z1}-{z2}, m:{m1}-{m2}, q:{q1}-{q2}, {filter_name}-band "
        f"gal_type={gal_type}, "
        f"re_img={re_img * pixel_to_kpc:.2f} kpc, "
        f"re_sbp_obs={re_sbp_obs * pixel_to_kpc:.2f} kpc, "
        f"re_sbp_deconv={re_sbp_deconv * pixel_to_kpc:.2f} kpc, "
        f"re_img_wiener={re_img_wiener * pixel_to_kpc:.2f} kpc, "
        f"re_sbp_wiener={re_sbp_wiener * pixel_to_kpc:.2f} kpc"
    )

    return {
        "z": avg_z,
        "mstar": midpoint(m1, m2),
        "q": midpoint(q1, q2),
        "gal_type": gal_type,
        "pixel_to_kpc": pixel_to_kpc,
        "filter": filter_name,
        "re_img": re_img,
        "re_sbp_obs": re_sbp_obs,
        "re_sbp_deconv": re_sbp_deconv,
        "re_img_wiener": re_img_wiener,
        "re_sbp_wiener": re_sbp_wiener,
    }


if __name__ == "__main__":
    # Define mass and redshift bins
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

    # Create list of all tasks to parallelize
    task_args = [
        (z1, z2, m1, m2, q1, q2, filter, gal_type)
        for z1, z2 in z_bins
        for m1, m2 in m_bins
        for q1, q2 in q_bins
        for filter in filters
        for gal_type in gal_types
    ]

    # Process all measurements in parallel
    print(
        f"Processing {len(task_args)} measurements using {min(len(task_args), ncores)} cores..."
    )
    with mp.Pool(processes=min(len(task_args), ncores)) as pool:
        results = pool.map(process_single_measurement, task_args)

    # Filter out None results and collect data
    results = [r for r in results if r is not None]
    results = append_restframe_i_rows(results)

    zarr = [r["z"] for r in results]
    marr = [r["mstar"] for r in results]
    qarr = [r["q"] for r in results]
    gal_type_arr = [r["gal_type"] for r in results]
    filter_arr = [r["filter"] for r in results]
    pixel_to_kpc_arr = [r["pixel_to_kpc"] for r in results]
    re_img_arr = [r["re_img"] for r in results]
    re_sbp_obs_arr = [r["re_sbp_obs"] for r in results]
    re_sbp_deconv_arr = [r["re_sbp_deconv"] for r in results]
    re_img_wiener_arr = [r["re_img_wiener"] for r in results]
    re_sbp_wiener_arr = [r["re_sbp_wiener"] for r in results]

    # assemble results into a table and save to disk
    result_table = Table(
        data={
            "z": zarr,
            "mstar": marr,
            "q": qarr,
            "gal_type": gal_type_arr,
            "filter": filter_arr,
            "pixel_to_kpc": pixel_to_kpc_arr,
            "re_img": re_img_arr,
            "re_sbp_obs": re_sbp_obs_arr,
            "re_sbp_deconv": re_sbp_deconv_arr,
            "re_img_wiener": re_img_wiener_arr,
            "re_sbp_wiener": re_sbp_wiener_arr,
        }
    )
    result_table["z"].info.format = ".2f"
    result_table["mstar"].info.format = ".2f"
    result_table["q"].info.format = ".2f"
    result_table.write(
        stack_dir / "hlr.txt", format="ascii.fixed_width", overwrite=True
    )
