#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=defq,hmemq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --time=1-00:00:00
#SBATCH --job-name=deconv_gals
#SBATCH --output=/gpfs01/home/ppzhg/logs/deconv_gals/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/deconv_gals/%j.err
# fmt: on

import argparse
import warnings
from pathlib import Path

import bottleneck as bn
import galsim
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from imcascade import Fitter
from skimage.restoration import unsupervised_wiener

cosmo = FlatLambdaCDM(
    H0=67.74 * u.km / u.s / u.Mpc,  # Hubble constant
    Om0=0.3089,  # Matter density parameter
    Ob0=0.04860,  # Baryon density parameter
    Tcmb0=2.7255 * u.K,  # CMB temperature
)

stack_dir = Path("~/Q1_gal_stacks_rot").expanduser()
psf_dir = Path("~/ero_psf").expanduser()
pixel_scale = 0.3
psf_oversamp = 3
psf_scale = pixel_scale / psf_oversamp


def avg_bkg_stack(img_path_base, nsky=100):
    imgs = []
    sky_stack_path = img_path_base.with_stem(img_path_base.stem + "_sky")
    with fits.open(sky_stack_path) as hdul:
        for i in range(nsky):
            imgs.append(np.asarray(hdul[i + 1].data, dtype=np.float64))
    imgs = np.asarray(imgs, dtype=np.float64)
    return bn.nanmean(imgs, axis=0)


def pad_image_centered(img, target_shape):
    """Zero-pad `img` to `target_shape`, centered."""
    Iy, Ix = target_shape
    ny, nx = img.shape
    out = np.zeros((Iy, Ix), dtype=img.dtype)

    # Starting index in padded array where the image will be placed
    y0 = (Iy - ny) // 2
    x0 = (Ix - nx) // 2

    out[y0 : y0 + ny, x0 : x0 + nx] = img
    return out


def crop_center(arr, shape):
    """Crop the center of `arr` to `shape`."""
    Iy, Ix = shape
    ny, nx = arr.shape
    y0 = (ny - Iy) // 2
    x0 = (nx - Ix) // 2
    return arr[y0 : y0 + Iy, x0 : x0 + Ix]


def deconvolve_image(
    gal_img_path,
    psf_img,
    sigma_psf,
    norm_psf,
    psf_fwhm_est,
    pixel_scale=0.3,
    bound_pix_rad=None,
):
    """
    Deconvolve a galaxy image using both imcascade and Wiener methods.

    Parameters
    ----------
    gal_img_path : Path
        Path to the galaxy image FITS file. The file should contain:
        - HDU 0: Galaxy image data
        - HDU 2: RMS/error map
    psf_img : numpy.ndarray
        The PSF image (normalized).
    sigma_psf : numpy.ndarray
        PSF sigma values from MGE decomposition.
    norm_psf : numpy.ndarray
        PSF normalization values from MGE decomposition.
    psf_fwhm_est : float
        Estimated PSF FWHM in arcsec.
    pixel_scale : float, optional
        Pixel scale in arcsec/pixel. Default is 0.3.
    bound_pix_rad : float, optional
        Radius in pixels to bound the image for parameter estimation.
        If None, uses the full image.

    Returns
    -------
    deconv_imcascade : numpy.ndarray
        Image deconvolved with imcascade method.
    deconv_wiener : numpy.ndarray
        Image deconvolved with Wiener method.
    """
    # Read galaxy image, mask, and RMS from FITS file and sanitize invalid pixels.
    with fits.open(gal_img_path) as hdul:
        gal_img = np.asarray(hdul[0].data, dtype=np.float64)
        gal_mask = np.zeros_like(gal_img, dtype=bool)
        if len(hdul) > 1 and hdul[1].data is not None:
            gal_mask |= np.asarray(hdul[1].data, dtype=bool)
        gal_rms = np.asarray(hdul[2].data, dtype=np.float64)

    bad_img = ~np.isfinite(gal_img)
    bad_rms = (~np.isfinite(gal_rms)) | (gal_rms <= 0)
    bad_pix = gal_mask | bad_img | bad_rms
    if np.any(bad_pix):
        print(
            "Masking"
            f" {bad_pix.sum()} invalid pixels"
            f" ({bad_img.sum()} bad image, {bad_rms.sum()} bad RMS)"
        )

    gal_img = gal_img.copy()
    gal_img[bad_pix] = 0.0
    weight = np.zeros_like(gal_rms, dtype=np.float64)
    good_pix = ~bad_pix
    weight[good_pix] = 1.0 / gal_rms[good_pix] ** 2
    if not np.any(weight > 0):
        raise ValueError(f"No valid pixels available for fitting in {gal_img_path}.")

    # estimate some galaxy parameters
    gal_img_ = galsim.fits.read(
        str(gal_img_path)
    )  # galsim is not compatible with pathlib!
    if bound_pix_rad is not None:
        xmid = (gal_img_.xmin + gal_img_.xmax) / 2
        ymid = (gal_img_.ymin + gal_img_.ymax) / 2
        rad = bound_pix_rad
        bounds = galsim.BoundsI(
            round(xmid - rad),
            round(xmid + rad),
            round(ymid - rad),
            round(ymid + rad),
        )
        gal_img_ = gal_img_.subImage(bounds)
    gal_img_ = galsim.InterpolatedImage(
        gal_img_,
        scale=pixel_scale,
        depixelize=False,
        normalization="flux",
    )
    gal_flux_est = gal_img_.flux
    gal_re_est = gal_img_.calculateHLR()
    gal_re_est_pix = gal_re_est / pixel_scale
    if gal_flux_est <= 0:
        raise ValueError("Estimated galaxy flux is non-positive.")
    print(f"Estimated galaxy flux: {gal_flux_est} within {bound_pix_rad} pixels")
    print(
        "Estimated galaxy effective radius:"
        f" {gal_re_est} arcsec ({gal_re_est_pix} pixels)"
    )

    # imcascade deconvolution
    sig = np.geomspace(
        psf_fwhm_est / pixel_scale / 2, gal_re_est / pixel_scale * 20, num=10
    )
    fitter = Fitter(
        gal_img,
        sig,
        sigma_psf,
        norm_psf,
        weight=weight,
        mask=bad_pix,
        sky_model=False,
        init_dict={
            # imcascade expects `re` in the same units as `sig`, i.e. pixels.
            "re": gal_re_est_pix,
            "flux": gal_flux_est,
            "q": 0.5 * (q1 + q2),
            # imcascade's internal +x is NumPy axis 0; the displayed horizontal
            # image axis is therefore its +y. phi=pi/2 puts the major axis along
            # the displayed horizontal axis for the untransposed FITS image.
            "phi": np.pi / 2,
        },
        # bounds_dict={"q": (0.999, 1.0)},
    )
    min_res = fitter.run_ls_min()
    best_fit_q = min_res[2]
    best_fit_phi = min_res[3]
    best_fit_phi_display = (best_fit_phi - np.pi / 2) % np.pi
    best_fit_pa_img_deg = np.degrees(best_fit_phi_display)
    best_fit_pa_xaxis_deg = min(best_fit_pa_img_deg, 180.0 - best_fit_pa_img_deg)
    if best_fit_pa_xaxis_deg > 10.0:
        warnings.warn(
            f"Best-fit PA is {best_fit_pa_xaxis_deg:.3f} deg from x-axis, exceeding 10 deg."
        )
    print(
        "imcascade best-fit shape:"
        f" q={best_fit_q:.6f},"
        f" PA={best_fit_pa_xaxis_deg:.3f} deg from x-axis"
    )
    fitter.save_results(
        gal_img_path.with_suffix(".asdf").with_stem(
            gal_img_path.stem + "_deconv_imcascade"
        )
    )
    # add back residuals to the model image
    conv_model_img = fitter.make_model(fitter.min_param)
    resid = gal_img - conv_model_img
    fitter.has_psf = False
    deconv_imcascade = fitter.make_model(fitter.min_param) + resid
    deconv_imcascade[bad_pix] = np.nan
    # generate a minimum header for the imcascade deconvolved image
    hdr = fits.Header()
    hdr["PSF_FWHM"] = (psf_fwhm_est, "PSF FWHM in arcsec")
    hdr["IC_Q"] = (best_fit_q, "imcascade best-fit axis ratio")
    hdr["IC_PA"] = (best_fit_pa_xaxis_deg, "imcascade PA in deg from x-axis")
    fits.writeto(
        gal_img_path.with_stem(gal_img_path.stem + "_deconv_imcascade"),
        deconv_imcascade,
        header=hdr,
        overwrite=True,
    )

    # Wiener deconvolution
    if gal_img.size < psf_img.size:
        gal_img_padded = pad_image_centered(gal_img, psf_img.shape)
        gal_deconv_padded, _ = unsupervised_wiener(
            gal_img_padded,
            psf_img,
            clip=False,
        )
        deconv_wiener = crop_center(gal_deconv_padded, gal_img.shape)
    else:
        deconv_wiener, _ = unsupervised_wiener(
            gal_img,
            psf_img,
            clip=False,
        )
    deconv_wiener[bad_pix] = np.nan
    # generate a minimum header for the wiener deconvolved image
    hdr = fits.Header()
    hdr["PSF_FWHM"] = (psf_fwhm_est, "PSF FWHM in arcsec")
    fits.writeto(
        gal_img_path.with_stem(gal_img_path.stem + "_deconv_wiener"),
        deconv_wiener,
        header=hdr,
        overwrite=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deconvolve the stacked galaxy image in a given redshift and stellar mass bin."
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
        help="Lower axis ratio limit (inclusive).",
    )
    parser.add_argument(
        "q2",
        type=float,
        help="Upper axis ratio limit.",
    )
    parser.add_argument(
        "filter",
        type=str,
        choices=["I", "Y", "J", "H"],
        help="Filter (I, Y, J, H)",
    )
    args = parser.parse_args()
    z1, z2, m1, m2, q1, q2, filter = (
        args.z1,
        args.z2,
        args.m1,
        args.m2,
        args.q1,
        args.q2,
        args.filter,
    )
    print(
        f"Deconvolving the stack image in z={z1}-{z2} and m={m1}-{m2} and q={q1}-{q2} in {filter} band..."
    )

    avg_z = 0.5 * (z1 + z2)
    angular_diameter_distance = cosmo.angular_diameter_distance(avg_z)
    pixel_scale_rad = (pixel_scale * u.arcsec).to(u.rad)
    pixel_to_mpc = pixel_scale_rad.value * angular_diameter_distance
    pixel_to_kpc = pixel_to_mpc.to(u.kpc).value
    if m1 >= 9.0 and m2 <= 10.0:
        bound_pix_rad = 100 / pixel_to_kpc
    if m1 >= 10.0 and m2 <= 11.0:
        bound_pix_rad = 200 / pixel_to_kpc
    if m1 >= 11.0 and m2 <= 12.0:
        bound_pix_rad = 500 / pixel_to_kpc

    # load PSF MGE and image
    psf_img_path = psf_dir / f"stack_1000_psf_{filter}_0.1.fits"
    psf_mge_path = psf_dir / f"stack_1000_psf_{filter}_0.1.txt"
    sigma_psf, norm_psf = np.loadtxt(psf_mge_path, unpack=True)
    sigma_psf = sigma_psf / psf_oversamp  # account for different pixel scales
    norm_psf = norm_psf / np.sum(norm_psf)  # normalize psf mge to unit flux

    # prepare low-res PSF image for Wiener deconvolution and estimate PSF FWHM
    psf_img_oversamp = galsim.fits.read(
        str(psf_img_path)
    )  # galsim is not compatible with pathlib!
    psf_hr = galsim.ImageF(psf_img_oversamp)
    psf_lr = psf_hr.bin(psf_oversamp, psf_oversamp)
    psf_img = psf_lr.array
    psf_img = psf_img / psf_img.sum()
    psf_img_oversamp = galsim.InterpolatedImage(
        psf_img_oversamp,
        scale=psf_scale,  # pixel scale of the PSF stamp
        depixelize=False,
        normalization="flux",
    )
    psf_fwhm_est = psf_img_oversamp.calculateFWHM()
    print(f"Estimated PSF FWHM: {psf_fwhm_est}")

    # deconvolve the LCG image
    lcg_img_path = (
        stack_dir / f"stack_lcg_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}_subbkg.fits"
    )
    if not lcg_img_path.exists():
        lcg_img_path_ = (
            stack_dir / f"stack_lcg_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}.fits"
        )
        if lcg_img_path_.exists():
            bkg = avg_bkg_stack(lcg_img_path_)
            with fits.open(lcg_img_path_) as hdul:
                lcg_gal_img = hdul[0].data
                lcg_gal_img -= bkg
                hdul.writeto(lcg_img_path)
    if lcg_img_path.exists():
        deconvolve_image(
            lcg_img_path,
            psf_img,
            sigma_psf,
            norm_psf,
            psf_fwhm_est,
            pixel_scale=pixel_scale,
            bound_pix_rad=bound_pix_rad,
        )

    # deconvolve the HCG image
    hcg_img_path = (
        stack_dir / f"stack_hcg_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}_subbkg.fits"
    )
    if not hcg_img_path.exists():
        hcg_img_path_ = (
            stack_dir / f"stack_hcg_{filter}_{z1}_{z2}_{m1}_{m2}_{q1}_{q2}.fits"
        )
        if hcg_img_path_.exists():
            bkg = avg_bkg_stack(hcg_img_path_)
            with fits.open(hcg_img_path_) as hdul:
                hcg_gal_img = hdul[0].data
                hcg_gal_img -= bkg
                hdul.writeto(hcg_img_path)
    if hcg_img_path.exists():
        deconvolve_image(
            hcg_img_path,
            psf_img,
            sigma_psf,
            norm_psf,
            psf_fwhm_est,
            pixel_scale=pixel_scale,
            bound_pix_rad=bound_pix_rad,
        )
