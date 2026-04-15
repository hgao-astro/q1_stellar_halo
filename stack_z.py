#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=hmemq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200g
#SBATCH --time=2-00:00:00
#SBATCH --job-name=stack_gals
#SBATCH --output=/gpfs01/home/ppzhg/logs/stack_gals/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/stack_gals/%j.err
# fmt: on

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

# Keep native math libraries single-threaded; this script parallelizes at Python level.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_var] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.io.fits import HDUList, PrimaryHDU
from astropy.io.votable import parse_single_table
from astropy.nddata import CCDData, Cutout2D, StdDevUncertainty
from astropy.nddata.utils import NoOverlapError
from astropy.stats import mad_std
from nicl.euclid.utilities import round_up_box_size
from photutils.background import Background2D
from skimage.transform import rotate

# Use Bottleneck's nanmedian if available (faster); else NumPy's.
try:
    import bottleneck as bn

    _nanmedian = bn.nanmedian
except Exception:
    _nanmedian = np.nanmedian

# Planck 2016 cosmology (Euclid standard)
cosmo = FlatLambdaCDM(
    H0=67.74 * u.km / u.s / u.Mpc,  # Hubble constant
    Om0=0.3089,  # Matter density parameter
    Ob0=0.04860,  # Baryon density parameter
    Tcmb0=2.7255 * u.K,  # CMB temperature
)
ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
pixel_scale = 0.3  # arcsec/pixel

# Global handle so forked workers can read the stack without copying on Linux.
_STACK = None


def _tile_nanmedian_mad(bounds):
    """Worker: median and MAD->sigma on a column stripe [x0:x1)."""
    x0, x1 = bounds
    blk = _STACK[:, :, x0:x1]  # (N, H, w)
    med, sig = nanmedian_mad(blk)
    return x0, med, sig


def _tile_nanmedian(bounds):
    """Worker: median on a column stripe [x0:x1)."""
    x0, x1 = bounds
    blk = _STACK[:, :, x0:x1]  # (N, H, w)
    med = nanmedian_image(blk)
    return x0, med


def nanmedian_image(stack):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        med = _nanmedian(stack, axis=0)
    return med


def nanmedian_mad(stack):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        med = _nanmedian(stack, axis=0)
        sig = mad_std(stack, axis=0, ignore_nan=True)
    return med, sig


def median_combine(
    arrays,
    nproc=None,
    dtype=np.float32,
    with_uncertainty=True,
    with_mask=True,
) -> CCDData:
    """
    Median-combine a list of 2D arrays (masked pixels already set to NaN),
    returning CCDData with robust per-pixel uncertainty (σ = 1.4826 * MAD).

    - Input: list of arrays, identical shape (H, W). NaNs mark invalid pixels.
    - Output: CCDData(data=median, mask=True where all inputs are NaN,
                      uncertainty=StdDevUncertainty(robust_sigma))
    """
    if not arrays:
        raise ValueError("No input arrays.")
    H, W = np.asarray(arrays[0]).shape
    if any(np.asarray(a).shape != (H, W) for a in arrays):
        raise ValueError("All input arrays must have identical shape.")

    out_dtype = np.dtype(dtype)
    work_dtype = np.float64

    # Build the working stack in float64 so bottleneck nanmedian always sees
    # the dtype that Astropy considers safe, then cast outputs as requested.
    stack = np.stack(
        [np.asarray(a, dtype=work_dtype, order="C") for a in arrays], axis=0
    )

    out_mask = None
    if with_mask:
        # Output mask: true where all inputs are NaN.
        out_mask = np.all(np.isnan(stack), axis=0)

    # Parallel tiling on X.
    nproc = min(nproc or (os.cpu_count() or 1), W)
    if nproc <= 1:
        # Single-process fallback
        if with_uncertainty:
            med, sig = nanmedian_mad(stack)
            med = med.astype(out_dtype, copy=False)
            sig = sig.astype(out_dtype, copy=False)
        else:
            med = nanmedian_image(stack).astype(out_dtype, copy=False)
            sig = None
    else:
        edges = np.linspace(0, W, nproc + 1, dtype=int)
        bounds = [
            (edges[i], edges[i + 1])
            for i in range(len(edges) - 1)
            if edges[i] < edges[i + 1]
        ]

        global _STACK
        _STACK = stack
        start = "fork" if sys.platform.startswith("linux") else "spawn"
        ctx = mp.get_context(start)
        try:
            with ctx.Pool(processes=len(bounds)) as pool:
                worker = _tile_nanmedian_mad if with_uncertainty else _tile_nanmedian
                parts = pool.map(worker, bounds)
        finally:
            _STACK = None

        med = np.empty((H, W), dtype=out_dtype)
        if with_uncertainty:
            sig = np.empty((H, W), dtype=out_dtype)
            for x0, mb, sb in parts:
                w = mb.shape[1]
                med[:, x0 : x0 + w] = mb
                sig[:, x0 : x0 + w] = sb
        else:
            sig = None
            for x0, mb in parts:
                w = mb.shape[1]
                med[:, x0 : x0 + w] = mb

    uncertainty = None
    if with_uncertainty:
        if out_mask is None:
            out_mask = np.isnan(med)
        # Set sigma=inf where everything was invalid
        sig = np.where(out_mask, np.inf, sig)
        uncertainty = StdDevUncertainty(sig)

    out = CCDData(med, unit="adu", mask=out_mask, uncertainty=uncertainty)
    out.meta["NCOMBINE"] = (len(arrays), "Number of frames in median combine")
    return out


def rotate_with_nan(
    img,
    angle_deg,
    weight_thresh=0.5,
):
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)

    x = np.where(finite, img, 0.0).astype(np.float32, copy=False)
    w = finite.astype(np.float32, copy=False)
    x_r = rotate(
        x,
        angle_deg,
        resize=False,
        order=1,
        mode="constant",
        cval=0.0,
        clip=False,
        preserve_range=True,
    )
    w_r = rotate(
        w,
        angle_deg,
        resize=False,
        order=1,
        mode="constant",
        cval=0.0,
        clip=False,
        preserve_range=True,
    )
    x_r = np.asarray(x_r, dtype=np.float32)
    w_r = np.asarray(w_r, dtype=np.float32)

    out = np.full_like(x_r, np.nan)
    good = w_r > np.float32(weight_thresh)
    np.divide(x_r, w_r, out=out, where=good)
    return out


def pa_to_x_axis_rotation(position_angle_deg):
    """Return the skimage rotation angle that aligns the major axis with +x."""
    # The catalog PA follows the usual north-style convention, i.e. it is
    # measured CCW from the image +y axis. In the image plane the major-axis
    # angleis therefore (90 - PA), so we rotate by its negative (PA - 90)
    # because `rotate` rotates clockwise (+x to -y) for positive angles.
    return float(position_angle_deg) - 90.0


def _rotate_and_scale_job(job):
    img, angle_deg, corr = job
    return rotate_with_nan(img, angle_deg=angle_deg) * corr


def rotate_and_scale_jobs(jobs, executor=None):
    if executor is None:
        return [_rotate_and_scale_job(job) for job in jobs]
    return list(executor.map(_rotate_and_scale_job, jobs))


def load_rotate_and_scale_jobs(jobs, executor=None):
    if executor is None:
        out = []
        for path, angle_deg, corr in jobs:
            img = fits.getdata(path, extname="SCI")
            out.append(_rotate_and_scale_job((img, angle_deg, corr)))
        return out

    max_pending = max(getattr(executor, "_max_workers", 1), 1)
    pending = deque()
    out = []
    for path, angle_deg, corr in jobs:
        img = fits.getdata(path, extname="SCI")
        pending.append(executor.submit(_rotate_and_scale_job, (img, angle_deg, corr)))
        if len(pending) >= max_pending:
            out.append(pending.popleft().result())
    while pending:
        out.append(pending.popleft().result())
    return out


def group_cutout_metadata_by_tile(tile_ids, corrs, rot_angles):
    grouped = {}
    for tile_id, corr, rot_angle in zip(tile_ids, corrs, rot_angles):
        group = grouped.setdefault(tile_id, {"corrs": [], "rot_angles": []})
        group["corrs"].append(corr)
        group["rot_angles"].append(rot_angle)
    for group in grouped.values():
        group["corrs"] = np.asarray(group["corrs"], dtype=np.float32)
        group["rot_angles"] = np.asarray(group["rot_angles"], dtype=np.float32)
    return grouped


def get_bkgsub_tile(
    tile_id,
    filter_name,
    tiles_dir,
    tile_segmap_dir,
    bkg_box_size,
    tiles_bkgsub,
    partial_tile_valid_pixels,
):
    tile_img = tiles_bkgsub.get(tile_id)
    if tile_img is None:
        if filter_name == "I":
            tile_path = tiles_dir / f"EUC_VIS_SWL-STK-{tile_id}.fits"
        else:
            tile_path = tiles_dir / f"EUC_NIR_W-STK_{filter_name}-{tile_id}.fits"
        vis_tile_segmap_path = (
            tile_segmap_dir / f"EUC_VIS_SWL-STK-{tile_id}_segmap.fits"
        )
        nir_tile_segmap_path = (
            tile_segmap_dir / f"EUC_NIR_W-STK_YJH-{tile_id}_segmap.fits"
        )
        with (
            fits.open(tile_path) as hdul,
            fits.open(vis_tile_segmap_path) as vis_segmap_hdul,
            fits.open(nir_tile_segmap_path) as nir_segmap_hdul,
        ):
            sci = hdul["SCI"].data
            vis_segmap = vis_segmap_hdul[1].data
            nir_segmap = nir_segmap_hdul[1].data
            sci_hdr = hdul["SCI"].header
            # perform background subtraction as in gal_cutout.py
            nx, ny = sci_hdr["NAXIS1"], sci_hdr["NAXIS2"]
            bkg_box_size_round = (
                round_up_box_size(ny, bkg_box_size),
                round_up_box_size(nx, bkg_box_size),
            )
            mask = (vis_segmap > 0) | (nir_segmap > 0) | np.isnan(sci)
            bg = Background2D(
                sci,
                bkg_box_size_round,
                mask=mask,
                exclude_percentile=80,
            )
            sci -= bg.background
            idx_valid = np.isfinite(sci)
            tile_img = np.where(mask, np.nan, sci).astype(np.float32)
            tiles_bkgsub[tile_id] = tile_img
            if idx_valid.sum() <= 0.2 * tile_img.size:
                partial_tile_valid_pixels[tile_id] = np.flatnonzero(idx_valid)
    return tile_img, partial_tile_valid_pixels.get(tile_id)


def draw_random_cutouts(tile_id, tile_img, valid_centers, n_cuts, rng, bkg_box_size):
    ny, nx = tile_img.shape
    cutout_imgs = []
    while len(cutout_imgs) < n_cuts:
        if valid_centers is None:
            cen_rand = rng.uniform((0, 0), (nx - 1, ny - 1))
        else:
            idx_cen = rng.integers(len(valid_centers))
            ceny, cenx = np.unravel_index(valid_centers[idx_cen], (ny, nx))
            cen_rand = (cenx, ceny)
        try:
            cutout_sci_rand = Cutout2D(
                tile_img,
                cen_rand,
                2 * bkg_box_size,
                mode="partial",
                fill_value=np.nan,
            )
        except NoOverlapError:
            print(f"No overlap for tile {tile_id}. This should not happen!")
            print(nx, ny, cen_rand, 2 * bkg_box_size)
            print(0 if valid_centers is None else len(valid_centers))
            sys.exit(1)
        data_cut = cutout_sci_rand.data
        if np.isfinite(data_cut).sum() <= 1:
            continue
        cutout_imgs.append(data_cut)
    return cutout_imgs


def extend_sky_stack_from_pool(
    imgs,
    tile_groups,
    cutout_pool,
    tile_pool_sizes,
    rng,
    filter_name,
    tiles_dir,
    tile_segmap_dir,
    bkg_box_size,
    tiles_bkgsub,
    partial_tile_valid_pixels,
    rot_executor=None,
):
    rot_jobs = []
    for tile_id, group in tile_groups.items():
        corrs = group["corrs"]
        rot_angles = group["rot_angles"]
        n_cuts = len(corrs)

        pool = cutout_pool.get(tile_id)
        if pool is None:
            tile_img, valid_centers = get_bkgsub_tile(
                tile_id,
                filter_name,
                tiles_dir,
                tile_segmap_dir,
                bkg_box_size,
                tiles_bkgsub,
                partial_tile_valid_pixels,
            )
            pool = draw_random_cutouts(
                tile_id,
                tile_img,
                valid_centers,
                tile_pool_sizes[tile_id],
                rng,
                bkg_box_size,
            )
            cutout_pool[tile_id] = pool

        if n_cuts == len(pool):
            selected_idx = np.arange(len(pool), dtype=int)
        else:
            selected_idx = rng.choice(len(pool), size=n_cuts, replace=False)

        for idx, corr, rot_angle in zip(selected_idx, corrs, rot_angles):
            rot_jobs.append((pool[int(idx)], rot_angle, corr))

    imgs.extend(rotate_and_scale_jobs(rot_jobs, executor=rot_executor))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine all galaxy images in a given redshift bin and filter."
    )
    parser.add_argument(
        "z1",
        type=float,
        help="Lower redshift limit (inclusive).",
    )
    parser.add_argument(
        "z2",
        type=float,
        help="Upper redshift limit.",
    )
    parser.add_argument(
        "filter",
        type=str,
        choices=["I", "Y", "J", "H"],
        help="Filter (I, Y, J, H)",
    )
    parser.add_argument(
        "--nsky",
        type=int,
        default=100,
        help="Number of random sky stacks.",
    )
    parser.add_argument(
        "--nbs",
        type=int,
        default=100,
        help="Number of bootstrap samples.",
    )
    parser.add_argument(
        "--bs_frac",
        type=float,
        default=1.0,
        help="Fraction of bootstrap samples to draw.",
    )
    parser.add_argument(
        "--min_gal_num",
        type=int,
        default=100,
        help="Minimum number of galaxies required to perform stacking.",
    )
    args = parser.parse_args()
    z1, z2, filter = args.z1, args.z2, args.filter
    nsky = args.nsky
    nbs = args.nbs
    bs_frac = args.bs_frac
    min_gal_num = args.min_gal_num
    if nsky < 0:
        print("nsky must be non-negative.")
        sys.exit(1)
    if nbs < 0:
        print("nbs must be non-negative.")
        sys.exit(1)
    if not (0 < bs_frac <= 1.0):
        print("bs_frac must be in (0, 1].")
        sys.exit(1)
    print(f"Stacking galaxies in {z1}<=z<{z2} in {filter}-band using {ncores} cores.")
    avg_redshift = 0.5 * (z1 + z2)
    stack_dir = Path("~/Q1_gal_stacks_rot").expanduser()
    tiles_dir = Path("~/Q1_tiles").expanduser()
    tile_segmap_dir = Path("~/Q1_tile_segmaps").expanduser()
    cutout_dir = Path("~/Q1_gal_cuts_combined_mask").expanduser()
    src_table = parse_single_table("~/catalogs/central_mass_cut_ext_pa.vot").to_table()
    rot_executor = ThreadPoolExecutor(max_workers=ncores) if ncores > 1 else None
    bin_records = []

    m_bins = [
        (9.0, 9.5),
        (9.5, 10.0),
        (10.0, 10.5),
        (10.5, 11.0),
        (11.0, 11.5),
        (11.5, 12.0),
    ]
    q_bins = [(0.0, 0.5), (0.5, 1.0), (0.0, 1.0)]
    # q_bins = [(0.0, 0.2), (0.2, 0.4), (0.0, 1.0)]
    mq_bins = list(product(m_bins, q_bins))
    for (m1, m2), (q1, q2) in mq_bins:
        print(
            f"Stacking galaxies in stellar mass bin [{m1}, {m2}) and axis ratio bin [{q1}, {q2})."
        )
        bin_record = {
            "bin": ((m1, m2), (q1, q2)),
            "lc_tile_ids": [],
            "lc_corrs": [],
            "lc_rot_angles": [],
            "hc_tile_ids": [],
            "hc_corrs": [],
            "hc_rot_angles": [],
        }
        # query for sources in the redshift bin and stellar mass bin
        selected_srcs = src_table[
            (src_table["photo_z"] >= z1)
            & (src_table["photo_z"] < z2)
            & (src_table["mstar"] >= m1)
            & (src_table["mstar"] < m2)
            & ((1 - src_table["ellipticity"]) > q1)
            & ((1 - src_table["ellipticity"]) <= q2)
        ]
        if len(selected_srcs) == 0:
            print(
                f"No sources found in redshift bin [{z1}, {z2}) and stellar mass bin [{m1}, {m2}) and axis ratio bin [{q1}, {q2}])."
            )
            bin_records.append(bin_record)
            continue
        print(f"Found {len(selected_srcs)} sources.")
        # select low and high concentration galaxies
        lc_idx = selected_srcs["sersic_index_vis"] < 2.5
        hc_idx = selected_srcs["sersic_index_vis"] >= 2.5
        lcgs = selected_srcs[lc_idx]
        hcgs = selected_srcs[hc_idx]

        # find lcg cutouts
        if len(lcgs) >= min_gal_num:
            lcg_obj_ids = lcgs["obj_id"]
            lcg_tile_ids = lcgs["tile_id"]
            lcg_extinctions = lcgs[f"ext_{filter.lower()}"]
            lcg_pas = lcgs["position_angle"]
            lcg_cutout_paths = []
            lcg_tile_ids_used = []
            lcg_corrections = []
            lcg_rot_angles = []
            for obj_id, tile_id, extinction, pa in zip(
                lcg_obj_ids, lcg_tile_ids, lcg_extinctions, lcg_pas
            ):
                if filter == "I":
                    path = cutout_dir / f"EUC_VIS_SWL-STK-{tile_id}_{obj_id}.fits"
                else:
                    path = (
                        cutout_dir / f"EUC_NIR_W-STK_{filter}-{tile_id}_{obj_id}.fits"
                    )
                if not path.exists():
                    print(f"Cutout not found for object {obj_id}. Skipping...")
                    continue
                lcg_cutout_paths.append(path)
                lcg_tile_ids_used.append(tile_id)
                lcg_corrections.append(
                    10 ** (0.4 * extinction)
                )  # to multiply the image
                lcg_rot_angles.append(pa_to_x_axis_rotation(pa))
            print(f"Found {len(lcg_cutout_paths)} LCG cutouts.")
            if len(lcg_cutout_paths) < len(lcgs):
                print("Cutouts for some LCGs are missing.")
            if len(lcg_cutout_paths) == 0:
                print("No cutouts found for LCG. Exiting...")
                sys.exit(0)
            # load galaxy cutouts
            lcg_jobs = list(zip(lcg_cutout_paths, lcg_rot_angles, lcg_corrections))
            lcg_imgs = load_rotate_and_scale_jobs(lcg_jobs, executor=rot_executor)
            time_start = time.time()
            lcg_stack_med = median_combine(lcg_imgs, nproc=ncores, dtype=np.float32)
            print(
                f"Median combining array {len(lcg_imgs)}x{lcg_imgs[0].shape[0]}x{lcg_imgs[0].shape[1]} took {time.time() - time_start:.1f} seconds."
            )
            lcg_stack_path = (
                stack_dir / f"stack_lcg_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}.fits"
            )
            lcg_stack_med.write(lcg_stack_path, overwrite=True)
            bin_record["lc_tile_ids"] = lcg_tile_ids_used
            bin_record["lc_corrs"] = lcg_corrections
            bin_record["lc_rot_angles"] = lcg_rot_angles
            if nbs > 0:
                print(
                    f"Generating {nbs} bootstrap samples with fraction {bs_frac} for LCGs..."
                )
                hdul = HDUList(PrimaryHDU())
                for i in range(nbs):
                    time_start_bs = time.time()
                    imgs_bs = random.choices(lcg_imgs, k=int(len(lcg_imgs) * bs_frac))
                    stack_med_bs = median_combine(
                        imgs_bs,
                        nproc=ncores,
                        dtype=np.float32,
                        with_uncertainty=False,
                        with_mask=False,
                    )
                    if i == 0 or (i + 1) % 10 == 0 or i + 1 == nbs:
                        print(
                            f"LCG bootstrap {i + 1}/{nbs} combine took {time.time() - time_start_bs:.1f} seconds."
                        )
                    lcg_hdu = stack_med_bs.to_hdu(as_image_hdu=True)[0]
                    lcg_hdu.header["extname"] = f"BS{i + 1}"
                    hdul.append(lcg_hdu)
                lcg_bs_stack_path = lcg_stack_path.with_stem(
                    lcg_stack_path.stem + "_bs"
                )
                hdul.writeto(lcg_bs_stack_path, overwrite=True)
        else:
            print(f"Only found {len(lcgs)} LCGs, skipping stacking.")
        # find hcg cutouts
        if len(hcgs) >= min_gal_num:
            hcg_obj_ids = hcgs["obj_id"]
            hcg_tile_ids = hcgs["tile_id"]
            hcg_extinctions = hcgs[f"ext_{filter.lower()}"]
            hcg_pas = hcgs["position_angle"]
            hcg_cutout_paths = []
            hcg_tile_ids_used = []
            hcg_corrections = []
            hcg_rot_angles = []
            for obj_id, tile_id, extinction, pa in zip(
                hcg_obj_ids, hcg_tile_ids, hcg_extinctions, hcg_pas
            ):
                if filter == "I":
                    path = cutout_dir / f"EUC_VIS_SWL-STK-{tile_id}_{obj_id}.fits"
                else:
                    path = (
                        cutout_dir / f"EUC_NIR_W-STK_{filter}-{tile_id}_{obj_id}.fits"
                    )
                if not path.exists():
                    print(f"Cutout not found for object {obj_id}. Skipping...")
                    continue
                hcg_cutout_paths.append(path)
                hcg_tile_ids_used.append(tile_id)
                hcg_rot_angles.append(pa_to_x_axis_rotation(pa))
                hcg_corrections.append(
                    10 ** (0.4 * extinction)
                )  # to multiply the image
            print(f"Found {len(hcg_cutout_paths)} HCG cutouts.")
            if len(hcg_cutout_paths) < len(hcgs):
                print("Cutouts for some HCGs are missing.")
            if len(hcg_cutout_paths) == 0:
                print("No cutouts found for HCG. Exiting...")
                sys.exit(0)
            # load galaxy cutouts
            hcg_jobs = list(zip(hcg_cutout_paths, hcg_rot_angles, hcg_corrections))
            hcg_imgs = load_rotate_and_scale_jobs(hcg_jobs, executor=rot_executor)
            time_start = time.time()
            hcg_stack_med = median_combine(hcg_imgs, nproc=ncores, dtype=np.float32)
            print(
                f"Median combining array {len(hcg_imgs)}x{hcg_imgs[0].shape[0]}x{hcg_imgs[0].shape[1]} took {time.time() - time_start:.1f} seconds."
            )
            hcg_stack_path = (
                stack_dir / f"stack_hcg_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}.fits"
            )
            hcg_stack_med.write(hcg_stack_path, overwrite=True)
            bin_record["hc_tile_ids"] = hcg_tile_ids_used
            bin_record["hc_corrs"] = hcg_corrections
            bin_record["hc_rot_angles"] = hcg_rot_angles
            if nbs > 0:
                print(
                    f"Generating {nbs} bootstrap samples with fraction {bs_frac} for HCGs..."
                )
                hdul = HDUList(PrimaryHDU())
                for i in range(nbs):
                    time_start_bs = time.time()
                    imgs_bs = random.choices(hcg_imgs, k=int(len(hcg_imgs) * bs_frac))
                    stack_med_bs = median_combine(
                        imgs_bs,
                        nproc=ncores,
                        dtype=np.float32,
                        with_uncertainty=False,
                        with_mask=False,
                    )
                    if i == 0 or (i + 1) % 10 == 0 or i + 1 == nbs:
                        print(
                            f"HCG bootstrap {i + 1}/{nbs} combine took {time.time() - time_start_bs:.1f} seconds."
                        )
                    hcg_hdu = stack_med_bs.to_hdu(as_image_hdu=True)[0]
                    hcg_hdu.header["extname"] = f"BS{i + 1}"
                    hdul.append(hcg_hdu)
                hcg_bs_stack_path = hcg_stack_path.with_stem(
                    hcg_stack_path.stem + "_bs"
                )
                hdul.writeto(hcg_bs_stack_path, overwrite=True)
        else:
            print(f"Only found {len(hcgs)} HCGs, skipping stacking.")
        bin_records.append(bin_record)

    # make random sky stacks
    if nsky == 0 or not any(
        record["lc_tile_ids"] or record["hc_tile_ids"] for record in bin_records
    ):
        print("No sky stacks to make. Exiting...")
        sys.exit(0)
    for record in bin_records:
        record["lc_hdul_sky"] = HDUList(PrimaryHDU())
        record["hc_hdul_sky"] = HDUList(PrimaryHDU())
    # sort based on len(tile_ids_used) so that mass bins with more galaxies get processed first
    bin_records = sorted(
        bin_records,
        key=lambda record: len(record["lc_tile_ids"]) + len(record["hc_tile_ids"]),
        reverse=True,
    )
    tile_pool_sizes = {}
    for record in bin_records:
        record["lc_by_tile"] = group_cutout_metadata_by_tile(
            record["lc_tile_ids"], record["lc_corrs"], record["lc_rot_angles"]
        )
        record["hc_by_tile"] = group_cutout_metadata_by_tile(
            record["hc_tile_ids"], record["hc_corrs"], record["hc_rot_angles"]
        )
        for tile_groups in (record["lc_by_tile"], record["hc_by_tile"]):
            for tile_id, group in tile_groups.items():
                tile_pool_sizes[tile_id] = max(
                    tile_pool_sizes.get(tile_id, 0), len(group["corrs"])
                )
    tiles_bkgsub = {}
    partial_tile_valid_pixels = {}
    cutout_radius = 1 * u.Mpc
    angular_size_rad = cutout_radius / cosmo.angular_diameter_distance(avg_redshift)
    angular_size_arcsec = (angular_size_rad * u.rad).to(u.arcsec)
    bkg_box_size = (angular_size_arcsec / (pixel_scale * u.arcsec)).value
    # initialize random number generator; do not set seed so that different runs produce different results
    rng = np.random.default_rng()
    try:
        for i in range(nsky):
            # regenerate the shared random-cutout pools for each iteration
            print(f"Generating sky stack iteration {i + 1}/{nsky}...")
            cutout_pool = {}
            # iterate over stellar mass bins again and reuse per-tile cutouts across all bins
            for record in bin_records:
                lcg_tile_groups = record["lc_by_tile"]
                lcg_hdul_sky = record["lc_hdul_sky"]
                hcg_tile_groups = record["hc_by_tile"]
                hcg_hdul_sky = record["hc_hdul_sky"]
                ((m1, m2), (q1, q2)) = record["bin"]
                # LCGs
                time_start = time.time()
                lcg_n_cutouts = sum(
                    len(group["corrs"]) for group in lcg_tile_groups.values()
                )
                print(
                    f"Stacking {lcg_n_cutouts} random cutouts from {len(lcg_tile_groups)} tiles for LCGs in stellar mass bin [{m1}, {m2}) and axis ratio bin [{q1}, {q2})..."
                )
                lcg_imgs = []
                extend_sky_stack_from_pool(
                    lcg_imgs,
                    lcg_tile_groups,
                    cutout_pool,
                    tile_pool_sizes,
                    rng,
                    filter,
                    tiles_dir,
                    tile_segmap_dir,
                    bkg_box_size,
                    tiles_bkgsub,
                    partial_tile_valid_pixels,
                    rot_executor=rot_executor,
                )
                print(
                    f"Iter {i + 1}, LCGs mass bin [{m1}, {m2}) and axis ratio bin [{q1}, {q2}]: extracting {len(lcg_imgs)} random cutouts took {time.time() - time_start:.1f} seconds."
                )
                if lcg_imgs:
                    time_start = time.time()
                    lcg_stack_med_sky = median_combine(
                        lcg_imgs,
                        nproc=ncores,
                        dtype=np.float32,
                        with_uncertainty=False,
                        with_mask=False,
                    )
                    print(
                        f"Iter {i + 1}, LCGs mass bin [{m1}, {m2}) and axis ratio bin [{q1}, {q2}]: median combining sky array {len(lcg_imgs)}x{lcg_imgs[0].shape[0]}x{lcg_imgs[0].shape[1]} took {time.time() - time_start:.1f} seconds."
                    )
                    lcg_hdu = lcg_stack_med_sky.to_hdu(as_image_hdu=True)[0]
                    lcg_hdu.header["extname"] = f"SKY{i + 1}"
                    lcg_hdul_sky.append(lcg_hdu)

                # HCGs
                time_start = time.time()
                hcg_n_cutouts = sum(
                    len(group["corrs"]) for group in hcg_tile_groups.values()
                )
                print(
                    f"Stacking {hcg_n_cutouts} random cutouts from {len(hcg_tile_groups)} tiles for HCGs in stellar mass bin [{m1}, {m2}) and axis ratio bin [{q1}, {q2}]..."
                )
                hcg_imgs = []
                extend_sky_stack_from_pool(
                    hcg_imgs,
                    hcg_tile_groups,
                    cutout_pool,
                    tile_pool_sizes,
                    rng,
                    filter,
                    tiles_dir,
                    tile_segmap_dir,
                    bkg_box_size,
                    tiles_bkgsub,
                    partial_tile_valid_pixels,
                    rot_executor=rot_executor,
                )
                print(
                    f"Iter {i + 1}, HCGs mass bin [{m1}, {m2}) and axis ratio bin [{q1}, {q2}]: extracting {len(hcg_imgs)} random cutouts took {time.time() - time_start:.1f} seconds."
                )
                if hcg_imgs:
                    time_start = time.time()
                    hcg_stack_med_sky = median_combine(
                        hcg_imgs,
                        nproc=ncores,
                        dtype=np.float32,
                        with_uncertainty=False,
                        with_mask=False,
                    )
                    print(
                        f"Iter {i + 1}, HCGs mass bin [{m1}, {m2}) and axis ratio bin [{q1}, {q2}]: median combining sky array {len(hcg_imgs)}x{hcg_imgs[0].shape[0]}x{hcg_imgs[0].shape[1]} took {time.time() - time_start:.1f} seconds."
                    )
                    hcg_hdu = hcg_stack_med_sky.to_hdu(as_image_hdu=True)[0]
                    hcg_hdu.header["extname"] = f"SKY{i + 1}"
                    hcg_hdul_sky.append(hcg_hdu)
    finally:
        if rot_executor is not None:
            rot_executor.shutdown()

    for record in bin_records:
        ((m1, m2), (q1, q2)) = record["bin"]
        hdul_sky_lcg = record["lc_hdul_sky"]
        hdul_sky_hcg = record["hc_hdul_sky"]
        lcg_stack_med_sky_path = (
            stack_dir / f"stack_lcg_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}_sky.fits"
        )
        hcg_stack_med_sky_path = (
            stack_dir / f"stack_hcg_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}_sky.fits"
        )
        if len(hdul_sky_lcg) > 1:
            hdul_sky_lcg.writeto(lcg_stack_med_sky_path, overwrite=True)
        if len(hdul_sky_hcg) > 1:
            hdul_sky_hcg.writeto(hcg_stack_med_sky_path, overwrite=True)
