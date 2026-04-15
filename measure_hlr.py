#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=shortq
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
from dataclasses import dataclass
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from imcascade import ImcascadeResults
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
mag_zpt = 23.9
ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


@dataclass
class SBP:
    """Container for observed surface brightness profile data"""

    radius: np.ndarray
    radius_kpc: np.ndarray
    sbp: np.ndarray
    sbp_err: np.ndarray


def intens_to_sb(intens, pixel_scale=0.3, mag_zpt=23.9):
    return mag_zpt - 2.5 * np.log10(intens / pixel_scale**2)


def get_sbps(z1, z2, m1, m2, q1, q2, filter, gal_type, nsigma=3):
    """
    Compute surface brightness profiles for original data and deconvolved images.
    """
    asdf_path = (
        stack_dir / f"sbps_{gal_type}_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}.asdf"
    )
    if asdf_path.exists():
        sbps_dict = asdf.open(asdf_path)
    else:
        print(f"No existing sbp file found at {asdf_path}.")
        return None, None, None
    pixel_to_kpc = sbps_dict.tree["pixel_to_kpc"]
    sbp_gal_table = sbps_dict.tree["gal"]
    sbp_sky_tables = sbps_dict.tree["sky"]
    sbp_bs_tables = sbps_dict.tree["bootstrap"]
    sbp_gal_deconv_imcascade_table = sbps_dict.tree["gal deconv imcascade"]
    sbp_gal_deconv_wiener_table = sbps_dict.tree["gal deconv wiener"]
    sma = sbp_gal_table["sma"]
    intens_gal = sbp_gal_table["intens"]
    intens_gal_deconv_imcascade = sbp_gal_deconv_imcascade_table["intens"]
    intens_gal_deconv_wiener = sbp_gal_deconv_wiener_table["intens"]
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
            # | (intens_gal_deconv_wiener < nsigma * comb_err)
        )[0][0]
    except IndexError:
        print(
            f"All data points are above threshold in {filter}-band, {z1}-{z2}, {m1}-{m2}."
        )
        rad_max_ind = len(sma)
    sma = sma[:rad_max_ind]
    sma_kpc = sma * pixel_to_kpc
    comb_err = comb_err[:rad_max_ind]
    intens_gal = intens_gal[:rad_max_ind]
    intens_gal_deconv_imcascade = intens_gal_deconv_imcascade[:rad_max_ind]
    intens_gal_deconv_wiener = intens_gal_deconv_wiener[:rad_max_ind]

    # compute imcascade mge profile (free of pixel response, no residuals added back)
    # res = ImcascadeResults(str(imcas_res_path))
    # sbp = res.calc_sbp(rad_for_imcascade)

    # Return organized results using dataclass
    obs_res = SBP(
        radius=sma,
        radius_kpc=sma_kpc,
        sbp=intens_gal,
        sbp_err=comb_err,
    )
    wiener_res = SBP(
        radius=sma,
        radius_kpc=sma_kpc,
        sbp=intens_gal_deconv_wiener,
        sbp_err=comb_err,
    )
    imcascade_res = SBP(
        radius=sma,
        radius_kpc=sma_kpc,
        sbp=intens_gal_deconv_imcascade,
        sbp_err=comb_err,
    )
    return obs_res, wiener_res, imcascade_res


def measure_re_from_img(img, radii=None):
    """Measure the half-light radius using curve of growth."""
    center = (img.shape[1] / 2 - 0.5, img.shape[0] / 2 - 0.5)
    if radii is None:
        radii = np.geomspace(1, img.shape[0] / 2 - 1, 50)
    cog = CurveOfGrowth(img, center, radii)
    cog.normalize()
    re = cog.calc_radius_at_ee(0.5)
    return re if re is not None else np.nan


def measure_re_from_sbp(sbp: SBP):
    """Measure the half-light radius by integrating the surface brightness profile."""
    # Convert intensity to flux per annulus
    intens = sbp.sbp  # in counts/s/pixel^2
    sma = sbp.radius  # in pixels

    # Calculate annulus boundaries
    # Each annulus i spans from sma[i] to sma[i+1]
    # Since sma[0] = 0, first annulus spans from 0 to sma[1]
    if len(sma) < 2:
        return np.nan

    # Inner edges: sma[i]
    r_inner = sma[:-1]
    # Outer edges: sma[i+1]
    r_outer = sma[1:]

    # Calculate annulus areas
    annulus_areas = np.pi * (r_outer**2 - r_inner**2)  # in pixels^2

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

    # Use PchipInterpolator to find radius at 50% enclosed energy
    try:
        interp = PchipInterpolator(fraction_valid, radii_valid, extrapolate=False)
        re = float(interp(0.5))
        return re
    except Exception:
        return np.nan


def measure_iso_radius(sbp, iso_sb=26.0):
    imcascade_sb = intens_to_sb(sbp.sbp)
    imcascade_radius = sbp.radius
    while np.any(np.diff(imcascade_sb) < 0):
        # select only the monotonic part
        valid_inds = np.where(np.diff(imcascade_sb) > 0)[0] + 1
        imcascade_sb = imcascade_sb[valid_inds]
        imcascade_radius = imcascade_radius[valid_inds]
    try:
        interp_sb = PchipInterpolator(imcascade_sb, imcascade_radius, extrapolate=True)
    except ValueError as e:
        print(np.array(imcascade_sb))
        print(np.array(imcascade_radius))
        raise e
    if float(interp_sb(iso_sb)) < 0:
        return np.max(imcascade_radius)
    return float(interp_sb(iso_sb))


def process_single_measurement(args):
    """Worker function to process a single (z, m, filter) combination."""
    z1, z2, m1, m2, q1, q2, filter, gal_type = args

    avg_z = 0.5 * (z1 + z2)
    angular_diameter_distance = cosmo.angular_diameter_distance(avg_z)
    pixel_scale_rad = (pixel_scale * u.arcsec).to(u.rad)
    pixel_to_mpc = pixel_scale_rad.value * angular_diameter_distance
    pixel_to_kpc = pixel_to_mpc.to(u.kpc).value

    deconv_asdf_path = (
        stack_dir
        / f"stack_{gal_type}_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}_subbkg_deconv_imcascade.asdf"
    )
    deconv_img_path = (
        stack_dir
        / f"stack_{gal_type}_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}_subbkg_deconv_imcascade.fits"
    )
    if not deconv_asdf_path.exists() or not deconv_img_path.exists():
        print(f"Deconvolved image/asdf not found: {deconv_asdf_path}")
        return None

    obs_sbp, _, imcascade_sbp = get_sbps(z1, z2, m1, m2, q1, q2, filter, gal_type)
    if obs_sbp is None or obs_sbp.radius.size < 5:
        print(
            f"Insufficient data points in sbp at z: {z1}-{z2}, m: {m1}-{m2}, q: {q1}-{q2}, {filter} band"
        )
        return None
    # find out radius where sbp drop to 26 mag arcsec^-2
    rad_26 = measure_iso_radius(imcascade_sbp, iso_sb=26.0)
    rad_29 = measure_iso_radius(imcascade_sbp, iso_sb=29.0)

    # re from multigaussian fit
    deconv_res = ImcascadeResults(str(deconv_asdf_path))
    re_imcascade = deconv_res.calc_r50(cutoff=imcascade_sbp.radius[-1])

    # re from integrating sbp
    re_sbp_obs = measure_re_from_sbp(obs_sbp)
    re_sbp_deconv = measure_re_from_sbp(imcascade_sbp)

    # re from curve of growth on image
    img = fits.getdata(deconv_img_path)
    re_img = measure_re_from_img(img, radii=imcascade_sbp.radius[1:])
    idx_26 = (imcascade_sbp.radius <= rad_26) & (imcascade_sbp.radius > 0)
    radii_26 = imcascade_sbp.radius[idx_26]
    radii_26 = np.append(radii_26, rad_26)
    re_img_26 = measure_re_from_img(img, radii=radii_26)

    print(
        f"z:{z1}-{z2}, m:{m1}-{m2}, q:{q1}-{q2}, {filter}-band "
        f"gal_type={gal_type}, "
        f"r26={rad_26 * pixel_to_kpc:.2f} kpc, "
        f"r29={rad_29 * pixel_to_kpc:.2f} kpc, "
        f"re_img={re_img * pixel_to_kpc:.2f} kpc, "
        f"re_img_26={re_img_26 * pixel_to_kpc:.2f} kpc, "
        f"re_sbp_obs={re_sbp_obs * pixel_to_kpc:.2f} kpc, "
        f"re_sbp_deconv={re_sbp_deconv * pixel_to_kpc:.2f} kpc, "
        f"re_imcascade={re_imcascade * pixel_to_kpc:.2f} kpc"
    )

    return {
        "z": avg_z,
        "mstar": 0.5 * (m1 + m2),
        "q": 0.5 * (q1 + q2),
        "q1": q1,
        "q2": q2,
        "gal_type": gal_type,
        "pixel_to_kpc": pixel_to_kpc,
        "filter": filter,
        "r26": rad_26,
        "r29": rad_29,
        "re_img": re_img,
        "re_img_26": re_img_26,
        "re_sbp_obs": re_sbp_obs,
        "re_sbp_deconv": re_sbp_deconv,
        "re_imcascade": re_imcascade,
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

    zarr = [r["z"] for r in results]
    marr = [r["mstar"] for r in results]
    qarr = [r["q"] for r in results]
    q1_arr = [r["q1"] for r in results]
    q2_arr = [r["q2"] for r in results]
    gal_type_arr = [r["gal_type"] for r in results]
    filter_arr = [r["filter"] for r in results]
    pixel_to_kpc_arr = [r["pixel_to_kpc"] for r in results]
    r26_arr = [r["r26"] for r in results]
    r29_arr = [r["r29"] for r in results]
    re_img_arr = [r["re_img"] for r in results]
    re_img_26_arr = [r["re_img_26"] for r in results]
    re_sbp_obs_arr = [r["re_sbp_obs"] for r in results]
    re_sbp_deconv_arr = [r["re_sbp_deconv"] for r in results]
    re_imcascade_arr = [r["re_imcascade"] for r in results]

    # assemble results into a table and save to disk
    result_table = Table(
        data={
            "z": zarr,
            "mstar": marr,
            "q": qarr,
            "q1": q1_arr,
            "q2": q2_arr,
            "gal_type": gal_type_arr,
            "filter": filter_arr,
            "pixel_to_kpc": pixel_to_kpc_arr,
            "r26": r26_arr,
            "r29": r29_arr,
            "re_img": re_img_arr,
            "re_img_26": re_img_26_arr,
            "re_sbp_obs": re_sbp_obs_arr,
            "re_sbp_deconv": re_sbp_deconv_arr,
            "re_imcascade": re_imcascade_arr,
        }
    )
    result_table.write(
        stack_dir / "hlr.txt", format="ascii.fixed_width", overwrite=True
    )
