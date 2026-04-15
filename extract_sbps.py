#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=defq
#SBATCH --mem=7g
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --job-name=extract_sbps
#SBATCH --output=/gpfs01/home/ppzhg/logs/extract_sbps/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/extract_sbps/%j.err
# fmt: on

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import asdf
import astropy.units as u
import bottleneck as bn
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from photutils.isophote import Ellipse, EllipseGeometry, Isophote, IsophoteList
from photutils.isophote.isophote import CentralPixel
from photutils.isophote.sample import CentralEllipseSample, EllipseSample

cosmo = FlatLambdaCDM(
    H0=67.74 * u.km / u.s / u.Mpc,  # Hubble constant
    Om0=0.3089,  # Matter density parameter
    Ob0=0.04860,  # Baryon density parameter
    Tcmb0=2.7255 * u.K,  # CMB temperature
)
stack_dir = Path("~/Q1_gal_stacks_rot").expanduser()
pixel_scale = 0.3  # arcsec/pixel
ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

FILTERS = ("I", "Y", "J", "H")
GAL_TYPES = ("lcg", "hcg")
REFERENCE_FILTER = "I"


def _normalize_sma_limits(img, minsma=None, maxsma=None):
    if minsma is not None and minsma < 0:
        raise ValueError("minsma must be non-negative or None.")
    maxsma_default = min(img.shape) / 2 - 1
    if maxsma is None:
        maxsma = maxsma_default
    else:
        maxsma = min(maxsma, maxsma_default)
    minsma = 0.5 if minsma is None else max(minsma, 0.5)
    if maxsma <= minsma:
        raise ValueError("maxsma must be larger than minsma.")
    return minsma, maxsma


def _default_sma0(minsma, maxsma, step, linear):
    if linear:
        sma0 = min(10.0, maxsma - step)
        sma0 = max(sma0, minsma + step)
    else:
        sma0 = min(10.0, 0.5 * (minsma + maxsma))
        sma0 = max(sma0, minsma * (1 + step))
    if sma0 >= maxsma:
        sma0 = 0.5 * (minsma + maxsma)
    return sma0


def _central_pixel_from_geometry(
    img,
    geometry,
    step=0.1,
    integrmode="bilinear",
    sclip=3,
    nclip=0,
    linear=False,
):
    sample0 = CentralEllipseSample(
        img,
        0.0,
        x0=geometry.x0,
        y0=geometry.y0,
        astep=step,
        eps=geometry.eps,
        position_angle=geometry.pa,
        sclip=sclip,
        nclip=nclip,
        linear_growth=linear,
        integrmode=integrmode,
    )
    sample0.update(geometry.fix)
    return CentralPixel(sample0)


def _prepend_central_pixel(
    img,
    isos,
    geometry,
    step=0.1,
    integrmode="bilinear",
    sclip=3,
    nclip=0,
    linear=False,
):
    iso_list = list(isos)
    if iso_list and np.isclose(iso_list[0].sample.geometry.sma, 0.0):
        return IsophoteList(iso_list)
    geometry0 = iso_list[0].sample.geometry if iso_list else geometry
    central_pixel = _central_pixel_from_geometry(
        img,
        geometry0,
        step=step,
        integrmode=integrmode,
        sclip=sclip,
        nclip=nclip,
        linear=linear,
    )
    return IsophoteList([central_pixel, *iso_list])


def _sample_isophotes(
    img,
    reference_isophotes,
    step=0.1,
    integrmode="bilinear",
    sclip=3,
    nclip=0,
    linear=False,
):
    if reference_isophotes is None:
        raise ValueError("reference_isophotes must be provided in sample mode.")
    isos = []
    for ref_iso in reference_isophotes:
        geometry = ref_iso.sample.geometry
        if np.isclose(geometry.sma, 0.0):
            isos.append(
                _central_pixel_from_geometry(
                    img,
                    geometry,
                    step=step,
                    integrmode=integrmode,
                    sclip=sclip,
                    nclip=nclip,
                    linear=linear,
                )
            )
            continue
        sample = EllipseSample(
            img,
            geometry.sma,
            x0=geometry.x0,
            y0=geometry.y0,
            astep=step,
            eps=geometry.eps,
            position_angle=geometry.pa,
            sclip=sclip,
            nclip=nclip,
            linear_growth=linear,
            integrmode=integrmode,
        )
        sample.update(geometry.fix)
        isos.append(Isophote(sample, 0, valid=True, stop_code=4))
    return IsophoteList(isos)


def extract_isophote(
    img,
    x0=None,
    y0=None,
    eps=0.0,
    pa=0.0,
    minsma=None,
    sma0=None,
    step=0.1,
    integrmode="bilinear",
    maxsma=None,
    sclip=3,
    nclip=0,
    linear=False,
    mode="sample",
    reference_isophotes=None,
    fix_center=True,
    fix_pa=False,
    fix_eps=False,
):
    if x0 is None:
        x0 = img.shape[1] / 2 - 0.5
    if y0 is None:
        y0 = img.shape[0] / 2 - 0.5

    minsma, maxsma = _normalize_sma_limits(img, minsma=minsma, maxsma=maxsma)

    if mode == "sample":
        return _sample_isophotes(
            img,
            reference_isophotes=reference_isophotes,
            step=step,
            integrmode=integrmode,
            sclip=sclip,
            nclip=nclip,
            linear=linear,
        )

    if mode != "fit":
        raise ValueError(f"Unsupported isophote extraction mode: {mode}")

    if sma0 is None:
        sma0 = _default_sma0(minsma, maxsma, step=step, linear=linear)
    geometry = EllipseGeometry(
        x0,
        y0,
        sma0,
        eps,
        pa,
        astep=step,
        linear_growth=linear,
        fix_center=fix_center,
        fix_pa=fix_pa,
        fix_eps=fix_eps,
    )
    ellipse = Ellipse(img, geometry)
    isos = ellipse.fit_image(
        sma0=sma0,
        minsma=minsma,
        maxsma=maxsma,
        step=step,
        sclip=sclip,
        nclip=nclip,
        integrmode=integrmode,
        linear=linear,
        fix_center=fix_center,
        fix_pa=fix_pa,
        fix_eps=fix_eps,
    )
    if len(isos) == 0:
        raise RuntimeError("Isophote fit failed to produce any valid isophotes.")
    return _prepend_central_pixel(
        img,
        isos,
        geometry,
        step=step,
        integrmode=integrmode,
        sclip=sclip,
        nclip=nclip,
        linear=linear,
    )


def fit_reference_isophotes(
    reference_img_path,
    maxsma,
    integrmode="median",
    sclip=3,
    nclip=10,
    step=0.1,
    linear=False,
):
    ref_img, ref_hdr = fits.getdata(reference_img_path, header=True)
    q0 = ref_hdr.get("IC_Q")
    eps0 = 0.0 if q0 is None else np.clip(1.0 - float(q0), 0.0, 0.95)
    pa0 = np.deg2rad(float(ref_hdr.get("IC_PA", 0.0)))
    return extract_isophote(
        ref_img,
        eps=eps0,
        pa=pa0,
        maxsma=maxsma,
        step=step,
        integrmode=integrmode,
        sclip=sclip,
        nclip=nclip,
        linear=linear,
        mode="fit",
        fix_center=True,
        fix_pa=False,
        fix_eps=False,
    )


def sample_image_to_table(
    img,
    reference_isophotes,
    integrmode="median",
    sclip=3,
    nclip=10,
    step=0.1,
    linear=False,
):
    isos = extract_isophote(
        img,
        step=step,
        integrmode=integrmode,
        sclip=sclip,
        nclip=nclip,
        linear=linear,
        mode="sample",
        reference_isophotes=reference_isophotes,
    )
    return fix_table_dtype(isos.to_table())


def process_stack(args):
    """Process a single image stack by sampling it on the reference isophotes."""
    img, reference_isophotes, integrmode, sclip, nclip, step, linear = args
    return sample_image_to_table(
        img,
        reference_isophotes=reference_isophotes,
        integrmode=integrmode,
        sclip=sclip,
        nclip=nclip,
        step=step,
        linear=linear,
    )


def sample_many_images(
    imgs,
    reference_isophotes,
    integrmode="median",
    sclip=3,
    nclip=10,
    step=0.1,
    linear=False,
    ncores=1,
):
    if len(imgs) == 0:
        return []
    args = [
        (img, reference_isophotes, integrmode, sclip, nclip, step, linear)
        for img in imgs
    ]
    if ncores <= 1 or len(args) == 1:
        return [process_stack(arg) for arg in args]
    with mp.Pool(processes=min(ncores, len(args))) as pool:
        return pool.map(process_stack, args)


def load_image_extensions(path, nimages):
    if nimages <= 0 or not path.exists():
        return []
    with fits.open(path) as hdul:
        nload = min(nimages, len(hdul) - 1)
        return [hdul[i + 1].data for i in range(nload)]


def fix_table_dtype(table, dtype=np.float64):
    """Convert all columns in an Astropy Table to a specified dtype."""
    for col in table.colnames:
        if table[col].dtype == object:
            table[col] = table[col].astype(dtype)
    return table


def build_stack_path(gal_type, filter_name, z1, z2, m1, m2, q1, q2):
    return (
        stack_dir / f"stack_{gal_type}_{filter_name}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}.fits"
    )


def build_sbp_path(gal_type, filter_name, z1, z2, m1, m2, q1, q2):
    return (
        stack_dir / f"sbps_{gal_type}_{filter_name}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}.asdf"
    )


def extract_all_sbps(
    gal_img_path,
    reference_isophotes,
    reference_img_path,
    output_path,
    integrmode="median",
    sclip=3,
    nclip=10,
    nsky=100,
    nbs=100,
    pixel_to_kpc=None,
    ncores=1,
    step=0.1,
    linear=False,
):
    """
    Extract surface brightness profiles from galaxy stacks and related images.

    All profiles are sampled on a common set of reference isophotes.
    """
    gal_img_subbkg_path = gal_img_path.with_stem(gal_img_path.stem + "_subbkg")
    gal_img_deconv_imcascade_path = gal_img_path.with_stem(
        gal_img_path.stem + "_subbkg_deconv_imcascade"
    )
    gal_img_deconv_wiener_path = gal_img_path.with_stem(
        gal_img_path.stem + "_subbkg_deconv_wiener"
    )
    required_paths = [
        gal_img_path,
        gal_img_subbkg_path,
        gal_img_deconv_imcascade_path,
        gal_img_deconv_wiener_path,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_str = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required image products: {missing_str}")

    sbp_dict = {
        "reference filter": REFERENCE_FILTER,
        "reference image": str(reference_img_path.name),
    }

    gal_img = fits.getdata(gal_img_path)
    gal_img_subbkg = fits.getdata(gal_img_subbkg_path)
    gal_img_deconv_imcascade = fits.getdata(gal_img_deconv_imcascade_path)
    gal_img_deconv_wiener = fits.getdata(gal_img_deconv_wiener_path)

    sbp_dict["gal+bkg"] = sample_image_to_table(
        gal_img,
        reference_isophotes=reference_isophotes,
        integrmode=integrmode,
        sclip=sclip,
        nclip=nclip,
        step=step,
        linear=linear,
    )
    sbp_dict["gal"] = sample_image_to_table(
        gal_img_subbkg,
        reference_isophotes=reference_isophotes,
        integrmode=integrmode,
        sclip=sclip,
        nclip=nclip,
        step=step,
        linear=linear,
    )

    sky_path = gal_img_path.with_stem(gal_img_path.stem + "_sky")
    sky_imgs = load_image_extensions(sky_path, nsky)
    if sky_imgs:
        bkg = bn.nanmean(sky_imgs, axis=0)
        sbp_dict["sky"] = sample_many_images(
            sky_imgs,
            reference_isophotes=reference_isophotes,
            integrmode=integrmode,
            sclip=sclip,
            nclip=nclip,
            step=step,
            linear=linear,
            ncores=ncores,
        )
    else:
        bkg = np.zeros_like(gal_img)
        sbp_dict["sky"] = []

    bs_path = gal_img_path.with_stem(gal_img_path.stem + "_bs")
    bs_imgs = load_image_extensions(bs_path, nbs)
    if bs_imgs:
        bs_imgs = [img - bkg for img in bs_imgs]
        sbp_dict["bootstrap"] = sample_many_images(
            bs_imgs,
            reference_isophotes=reference_isophotes,
            integrmode=integrmode,
            sclip=sclip,
            nclip=nclip,
            step=step,
            linear=linear,
            ncores=ncores,
        )
    else:
        sbp_dict["bootstrap"] = []

    if gal_img_deconv_imcascade_path == reference_img_path:
        sbp_dict["gal deconv imcascade"] = fix_table_dtype(
            reference_isophotes.to_table()
        )
    else:
        sbp_dict["gal deconv imcascade"] = sample_image_to_table(
            gal_img_deconv_imcascade,
            reference_isophotes=reference_isophotes,
            integrmode=integrmode,
            sclip=sclip,
            nclip=nclip,
            step=step,
            linear=linear,
        )
    sbp_dict["gal deconv wiener"] = sample_image_to_table(
        gal_img_deconv_wiener,
        reference_isophotes=reference_isophotes,
        integrmode=integrmode,
        sclip=sclip,
        nclip=nclip,
        step=step,
        linear=linear,
    )

    if pixel_to_kpc is not None:
        sbp_dict["pixel_to_kpc"] = pixel_to_kpc

    af = asdf.AsdfFile(sbp_dict)
    af.write_to(output_path, all_array_compression="zlib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract surface-brightness profiles for all filters in one q-bin."
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
        "m1",
        type=float,
        help="Lower stellar mass limit (inclusive).",
    )
    parser.add_argument(
        "m2",
        type=float,
        help="Upper stellar mass limit.",
    )
    parser.add_argument(
        "q1",
        type=float,
        help="Lower axis-ratio limit (exclusive).",
    )
    parser.add_argument(
        "q2",
        type=float,
        help="Upper axis-ratio limit (inclusive).",
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
        help="Number of random bootstrap samples.",
    )

    args = parser.parse_args()
    z1, z2, m1, m2, q1, q2 = args.z1, args.z2, args.m1, args.m2, args.q1, args.q2
    nsky = args.nsky
    nbs = args.nbs

    avg_z = 0.5 * (z1 + z2)
    angular_diameter_distance = cosmo.angular_diameter_distance(avg_z)
    pixel_scale_rad = (pixel_scale * u.arcsec).to(u.rad)
    pixel_to_mpc = pixel_scale_rad.value * angular_diameter_distance
    pixel_to_kpc = pixel_to_mpc.to(u.kpc).value
    maxsma = 1000 / pixel_to_kpc
    integrmode = "median"
    sclip = 3
    nclip = 10
    step = 0.1
    linear = False

    for gal_type in GAL_TYPES:
        reference_base_path = build_stack_path(
            gal_type,
            REFERENCE_FILTER,
            z1,
            z2,
            m1,
            m2,
            q1,
            q2,
        )
        reference_img_path = reference_base_path.with_stem(
            reference_base_path.stem + "_subbkg_deconv_imcascade"
        )
        if not reference_img_path.exists():
            print(
                f"Missing H-band reference image for {gal_type}: {reference_img_path}. "
                "Skipping this galaxy type."
            )
            continue

        print(f"Fitting H-band reference isophotes from {reference_img_path}")
        reference_isophotes = fit_reference_isophotes(
            reference_img_path,
            maxsma=maxsma,
            integrmode=integrmode,
            sclip=sclip,
            nclip=nclip,
            step=step,
            linear=linear,
        )

        for filter_name in FILTERS:
            gal_img_path = build_stack_path(
                gal_type, filter_name, z1, z2, m1, m2, q1, q2
            )
            output_path = build_sbp_path(gal_type, filter_name, z1, z2, m1, m2, q1, q2)
            if not gal_img_path.exists():
                print(f"Skipping missing stack: {gal_img_path}")
                continue
            print(
                f"Extracting SBPs for {gal_type} in {filter_name} band using "
                f"H-band reference isophotes."
            )
            try:
                extract_all_sbps(
                    gal_img_path,
                    reference_isophotes=reference_isophotes,
                    reference_img_path=reference_img_path,
                    output_path=output_path,
                    integrmode=integrmode,
                    sclip=sclip,
                    nclip=nclip,
                    nsky=nsky,
                    nbs=nbs,
                    pixel_to_kpc=pixel_to_kpc,
                    ncores=ncores,
                    step=step,
                    linear=linear,
                )
            except FileNotFoundError as exc:
                print(
                    f"Skipping {gal_type} {filter_name}-band due to missing inputs: {exc}"
                )
