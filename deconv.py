#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=defq,hmemq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10g
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
from skimage.restoration import richardson_lucy, unsupervised_wiener

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
DECONV_SUFFIXES = {
    "imcascade": "_deconv_imcascade",
    "wiener": "_deconv_wiener",
    "richardson_lucy": "_deconv_richardson_lucy",
    "pysersic": "_deconv_pysersic",
}


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


def get_center_crop_slices(shape, half_size):
    """Return slices for a centered square cutout."""
    if half_size is None:
        return slice(0, shape[0]), slice(0, shape[1])
    ny, nx = shape
    max_half_size = max(1, min(ny, nx) // 2 - 1)
    half_size = int(np.clip(np.round(half_size), 1, max_half_size))
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    x0 = max(0, int(np.floor(cx - half_size)))
    x1 = min(nx, int(np.ceil(cx + half_size + 1)))
    y0 = max(0, int(np.floor(cy - half_size)))
    y1 = min(ny, int(np.ceil(cy + half_size + 1)))
    return slice(y0, y1), slice(x0, x1)


def pad_image_to_min_shape(img, min_shape):
    """Pad `img` to at least `min_shape`, keeping the image centered."""
    target_shape = tuple(max(cur, req) for cur, req in zip(img.shape, min_shape))
    if target_shape == img.shape:
        return img
    return pad_image_centered(img, target_shape)


def run_wiener(img, psf_img):
    """Run unsupervised Wiener deconvolution, padding if needed."""
    padded_img = pad_image_to_min_shape(img, psf_img.shape)
    deconv_padded, _ = unsupervised_wiener(
        padded_img,
        psf_img,
        clip=False,
    )
    return crop_center(deconv_padded, img.shape)


def run_richardson_lucy(img, psf_img, num_iter):
    """Run Richardson-Lucy on a non-negative image, padding if needed."""
    rl_input = np.clip(img, 0.0, None)
    padded_img = pad_image_to_min_shape(rl_input, psf_img.shape)
    deconv_padded = richardson_lucy(
        padded_img,
        psf_img,
        num_iter=num_iter,
        clip=False,
    )
    return crop_center(deconv_padded, img.shape)


def make_delta_psf(psf_img):
    """Return a delta-function PSF on the same grid as `psf_img`."""
    delta_psf = np.zeros_like(psf_img, dtype=np.float32)
    cy, cx = np.asarray(psf_img.shape) // 2
    delta_psf[cy, cx] = 1.0
    return delta_psf


def crop_psf_for_fit(psf_img, image_shape):
    """Center-crop the PSF to the largest legal stamp for `image_shape`."""
    psf_img = np.asarray(psf_img, dtype=np.float32)
    psf_img = psf_img / np.sum(psf_img)
    cy, cx = np.asarray(psf_img.shape) // 2
    max_half_size = max(1, min(image_shape) // 2 - 1)
    max_psf_half_size = min(
        cy, cx, psf_img.shape[0] - cy - 1, psf_img.shape[1] - cx - 1
    )
    half_size = min(max_half_size, max_psf_half_size)

    y0 = cy - half_size
    y1 = cy + half_size + 1
    x0 = cx - half_size
    x1 = cx + half_size + 1
    cropped = psf_img[y0:y1, x0:x1].copy()
    cropped /= np.sum(cropped)
    return cropped


def run_pysersic_map(
    fitter,
    rkey,
    init_values=None,
    max_train=500,
    patience=75,
    num_round=2,
    lr_init=0.05,
    frac_lr_decrease=0.1,
):
    """Run pysersic's MAP optimizer with adjustable early-stopping settings."""
    from numpyro import infer, optim
    from numpyro.handlers import condition, trace
    from numpyro.infer import SVI, Trace_ELBO
    from pysersic.pysersic import train_numpyro_svi_early_stop

    model_cur = fitter.build_model(return_model=True)
    if init_values is None:
        init_values = {}
    init_loc_fn = infer.init_to_value(values=init_values)

    autoguide_map = infer.autoguide.AutoDelta(
        model_cur,
        init_loc_fn=init_loc_fn,
    )
    svi_kernel = SVI(model_cur, autoguide_map, optim.Adam(0.01), loss=Trace_ELBO())
    res = train_numpyro_svi_early_stop(
        svi_kernel,
        rkey=rkey,
        lr_init=lr_init,
        num_round=num_round,
        frac_lr_decrease=frac_lr_decrease,
        patience=patience,
        optimizer=optim.Adam,
        max_train=max_train,
    )

    use_dict = {}
    for key in res.params.keys():
        pref = key.split("_auto_loc")[0]
        use_dict[pref] = res.params[key]

    trace_out = trace(condition(model_cur, use_dict)).get_trace()
    real_out = {}
    for key in trace_out:
        if "Loss" in key:
            continue
        if key == "model":
            real_out[key] = np.asarray(trace_out[key]["value"])
        elif not (
            "base" in key
            or "auto" in key
            or "unwrapped" in key
            or "factor" in key
            or "loss" in key
        ):
            real_out[key] = np.round(trace_out[key]["value"], 5)
    return real_out


def write_deconvolved_image(gal_img_path, method, img, psf_fwhm_est, extra_header=None):
    """Write the deconvolved image for a single method."""
    hdr = fits.Header()
    hdr["PSF_FWHM"] = (psf_fwhm_est, "PSF FWHM in arcsec")
    hdr["DECMETH"] = (method, "Deconvolution method")
    if extra_header is not None:
        for key, value in extra_header.items():
            hdr[key] = value
    fits.writeto(
        gal_img_path.with_stem(gal_img_path.stem + DECONV_SUFFIXES[method]),
        img,
        header=hdr,
        overwrite=True,
    )


def estimate_galaxy_parameters(gal_img_path, pixel_scale=0.3, bound_pix_rad=None):
    """Estimate galaxy flux and effective radius for imcascade initialization."""
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
    return gal_flux_est, gal_re_est, gal_re_est_pix


def run_pysersic_doublesersic(
    gal_img,
    gal_rms,
    bad_pix,
    psf_img,
    theta_max_deg=10.0,
    theta_sigma_deg=5.0,
    inner_q_bounds=(0.2, 1.0),
    inner_q_init=0.6,
    outer_q_bounds=(0.2, 1.0),
    outer_q_init=0.8,
    max_train=500,
    patience=75,
    num_round=2,
    center_guess=None,
    render_shape=None,
    render_origin=(0, 0),
):
    """
    Fit a background-subtracted image with a pysersic doublesersic model.

    This follows the pysersic single-source and multi-profile docs:
    - use `SourceProperties(...).generate_prior("doublesersic", sky_type="none")`
    - fit with `FitSingle(...).find_MAP(...)`
    - render an intrinsic model separately by swapping in a delta-function PSF
    """
    try:
        import jax.numpy as jnp
        from jax.random import PRNGKey
        from pysersic import FitSingle, check_input_data
        from pysersic.loss import student_t_loss
        from pysersic.priors import SourceProperties
        from pysersic.rendering import HybridRenderer
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "The `pysersic` deconvolution path requires a compatible `pysersic`"
            " runtime. The tested combination here is `pysersic==0.1.5` with"
            " `arviz<1`, plus its `jax` dependencies."
        ) from exc

    fit_img = np.asarray(gal_img, dtype=np.float32)
    fit_mask = np.asarray(bad_pix, dtype=bool)
    fit_rms = np.asarray(gal_rms, dtype=np.float32)
    if render_shape is None:
        render_shape = fit_img.shape
    good_rms = fit_rms[~fit_mask & np.isfinite(fit_rms) & (fit_rms > 0)]
    if good_rms.size == 0:
        raise ValueError("No positive RMS values available for pysersic fitting.")
    fill_rms = np.float32(np.nanmedian(good_rms))
    fit_rms = np.where(
        np.isfinite(fit_rms) & (fit_rms > 0),
        fit_rms,
        fill_rms,
    )
    fit_rms[fit_mask] = fill_rms
    fit_psf = crop_psf_for_fit(psf_img, fit_img.shape)
    render_psf = crop_psf_for_fit(psf_img, render_shape)
    check_input_data(data=fit_img, rms=fit_rms, psf=fit_psf, mask=fit_mask)

    # SourceProperties uses photutils-derived source estimates. Suppress
    # negative sky fluctuations only for prior construction; the actual fit uses
    # the original background-subtracted image.
    prior_img = np.clip(fit_img, 0.0, None)
    props = SourceProperties(image=prior_img, mask=fit_mask)
    if center_guess is not None:
        props.set_position_guess(center_guess)
    props.set_theta_guess(0.0)
    print(
        "pysersic SourceProperties:"
        f" xc={props.xc_guess:.3f},"
        f" yc={props.yc_guess:.3f},"
        f" re={props.r_eff_guess:.3f} pix,"
        f" flux={props.flux_guess:.3f}"
    )
    prior = props.generate_prior("doublesersic", sky_type="none")

    inner_q_low, inner_q_high = np.sort(np.asarray(inner_q_bounds, dtype=np.float64))
    inner_q_low = np.clip(inner_q_low, 0.1, 1.0)
    inner_q_high = np.clip(inner_q_high, inner_q_low, 1.0)
    if inner_q_high <= inner_q_low:
        inner_q_low = max(0.1, inner_q_low - 1e-3)
        inner_q_high = min(1.0, inner_q_low + 1e-3)
    outer_q_low, outer_q_high = np.sort(np.asarray(outer_q_bounds, dtype=np.float64))
    outer_q_low = np.clip(outer_q_low, 0.1, 1.0)
    outer_q_high = np.clip(outer_q_high, outer_q_low, 1.0)
    if outer_q_high <= outer_q_low:
        outer_q_low = max(0.1, outer_q_low - 1e-3)
        outer_q_high = min(1.0, outer_q_low + 1e-3)
    inner_q_init = float(np.clip(inner_q_init, inner_q_low, inner_q_high))
    outer_q_init = float(np.clip(outer_q_init, outer_q_low, outer_q_high))

    if hasattr(prior, "set_truncated_gaussian_prior"):
        prior.set_truncated_gaussian_prior(
            "theta",
            0.0,
            np.deg2rad(theta_sigma_deg),
            low=-np.deg2rad(theta_max_deg),
            high=np.deg2rad(theta_max_deg),
        )
    if hasattr(prior, "set_uniform_prior"):
        prior.set_uniform_prior("ellip_1", 1.0 - inner_q_high, 1.0 - inner_q_low)
        prior.set_uniform_prior("ellip_2", 1.0 - outer_q_high, 1.0 - outer_q_low)

    ny, nx = fit_img.shape
    xc_default = 0.5 * (nx - 1)
    yc_default = 0.5 * (ny - 1)
    if center_guess is None:
        xc_init = xc_default
        yc_init = yc_default
    else:
        xc_init = float(center_guess[0])
        yc_init = float(center_guess[1])

    r_eff_init = float(max(0.5, props.r_eff_guess))
    init_values = {
        "theta": 0.0,
        "ellip_1": 1.0 - inner_q_init,
        "ellip_2": 1.0 - outer_q_init,
        "xc": xc_init,
        "yc": yc_init,
        "flux": float(max(1e-6, props.flux_guess)),
        "f_1": 0.5,
        "r_eff_1": float(max(0.5, r_eff_init / 1.5)),
        "r_eff_2": float(max(0.5, r_eff_init * 1.5)),
        "n_1": 4.0,
        "n_2": 1.0,
    }
    print(
        "pysersic priors:"
        f" theta~N(0,{theta_sigma_deg:.1f} deg)"
        f" truncated to +/-{theta_max_deg:.1f} deg,"
        f" q_inner in [{inner_q_low:.3f}, {inner_q_high:.3f}],"
        f" q_outer in [{outer_q_low:.3f}, {outer_q_high:.3f}]"
    )
    print(
        "pysersic init:"
        f" q_inner={inner_q_init:.3f},"
        f" q_outer={outer_q_init:.3f},"
        f" xc={xc_init:.3f},"
        f" yc={yc_init:.3f},"
        f" flux={init_values['flux']:.3f},"
        f" f1={init_values['f_1']:.3f},"
        f" re1={init_values['r_eff_1']:.3f},"
        f" re2={init_values['r_eff_2']:.3f},"
        f" n1={init_values['n_1']:.3f},"
        f" n2={init_values['n_2']:.3f}"
    )

    fitter = FitSingle(
        data=fit_img,
        rms=fit_rms,
        mask=fit_mask,
        psf=fit_psf,
        prior=prior,
        loss_func=student_t_loss,
    )
    print(
        "Running pysersic MAP:"
        f" max_train={max_train}, patience={patience}, num_round={num_round}"
    )
    map_params = run_pysersic_map(
        fitter,
        rkey=PRNGKey(1000),
        init_values=init_values,
        max_train=max_train,
        patience=patience,
        num_round=num_round,
    )
    fit_params = {
        key: np.asarray(value).item() if np.asarray(value).shape == () else value
        for key, value in map_params.items()
        if key != "model"
    }
    x_origin, y_origin = render_origin
    fit_params["xc"] = float(fit_params["xc"]) + float(x_origin)
    fit_params["yc"] = float(fit_params["yc"]) + float(y_origin)

    conv_renderer = HybridRenderer(
        render_shape,
        jnp.array(render_psf.astype(np.float32)),
    )
    conv_model_img = np.asarray(
        conv_renderer.render_source(fit_params, profile_type="doublesersic"),
        dtype=np.float64,
    )
    intrinsic_renderer = HybridRenderer(
        render_shape,
        jnp.array(make_delta_psf(render_psf)),
    )
    intrinsic_model_img = np.asarray(
        intrinsic_renderer.render_source(fit_params, profile_type="doublesersic"),
        dtype=np.float64,
    )
    return intrinsic_model_img, conv_model_img, fit_params


def deconvolve_image(
    gal_img_path,
    psf_img,
    sigma_psf,
    norm_psf,
    psf_fwhm_est,
    method,
    axis_ratio_init,
    axis_ratio_bounds=None,
    pixel_scale=0.3,
    bound_pix_rad=None,
    rl_num_iter=4,
    pysersic_max_train=500,
    pysersic_patience=75,
    pysersic_num_round=2,
):
    """
    Deconvolve a galaxy image with a single specified method.

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
    method : {"imcascade", "wiener", "richardson_lucy", "pysersic"}
        The single deconvolution method to apply.
    axis_ratio_init : float
        Initial axis ratio used by imcascade and as the inner q initialization for
        pysersic.
    axis_ratio_bounds : tuple[float, float], optional
        Axis-ratio range from the input bin, used as the inner q prior for pysersic.
    pixel_scale : float, optional
        Pixel scale in arcsec/pixel. Default is 0.3.
    bound_pix_rad : float, optional
        Radius in pixels to bound the image for parameter estimation.
        If None, uses the full image.
    rl_num_iter : int, optional
        Number of Richardson-Lucy iterations when that method is used.
    pysersic_max_train : int, optional
        Maximum training steps per round for pysersic MAP fitting.
    pysersic_patience : int, optional
        Early-stopping patience per round for pysersic MAP fitting.
    pysersic_num_round : int, optional
        Number of learning-rate rounds for pysersic MAP fitting.
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

    if method == "imcascade":
        gal_flux_est, gal_re_est, gal_re_est_pix = estimate_galaxy_parameters(
            gal_img_path,
            pixel_scale=pixel_scale,
            bound_pix_rad=bound_pix_rad,
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
                "q": axis_ratio_init,
                # imcascade's internal +x is NumPy axis 0; the displayed horizontal
                # image axis is therefore its +y. phi=pi/2 puts the major axis along
                # the displayed horizontal axis for the untransposed FITS image.
                "phi": np.pi / 2,
            },
            bounds_dict={
                "phi": (np.pi / 2 - np.deg2rad(10), np.pi / 2 + np.deg2rad(10)),
                "q": (0.2, 1.0),
            },
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
        deconv_img = fitter.make_model(fitter.min_param) + resid
        deconv_img[bad_pix] = np.nan
        write_deconvolved_image(
            gal_img_path,
            method,
            deconv_img,
            psf_fwhm_est,
            extra_header={
                "IC_Q": (best_fit_q, "imcascade best-fit axis ratio"),
                "IC_PA": (best_fit_pa_xaxis_deg, "imcascade PA in deg from x-axis"),
            },
        )
        return

    if method == "wiener":
        deconv_img = run_wiener(gal_img, psf_img)
        deconv_img[bad_pix] = np.nan
        write_deconvolved_image(
            gal_img_path,
            method,
            deconv_img,
            psf_fwhm_est,
        )
        return

    if method == "richardson_lucy":
        clipped_input_count = int(np.sum(gal_img < 0))
        if clipped_input_count > 0:
            print(
                "Clipping"
                f" {clipped_input_count} negative input pixels to zero before"
                " Richardson-Lucy."
            )
        deconv_img = run_richardson_lucy(gal_img, psf_img, rl_num_iter)
        deconv_img[bad_pix] = np.nan
        write_deconvolved_image(
            gal_img_path,
            method,
            deconv_img,
            psf_fwhm_est,
            extra_header={
                "DECITER": (rl_num_iter, "RL iterations"),
                "DECLIP": (clipped_input_count > 0, "Negative input clipped to zero"),
            },
        )
        return

    if method == "pysersic":
        if axis_ratio_bounds is None:
            raise ValueError("pysersic requires `axis_ratio_bounds` to be provided.")
        fit_img = gal_img
        fit_rms = gal_rms
        fit_bad_pix = bad_pix.copy()
        fit_origin_x = 0
        fit_origin_y = 0
        if bound_pix_rad is not None:
            fit_half_size = float(bound_pix_rad)
            y_slice, x_slice = get_center_crop_slices(gal_img.shape, fit_half_size)
            fit_origin_y = y_slice.start
            fit_origin_x = x_slice.start
            fit_img = gal_img[y_slice, x_slice]
            fit_rms = gal_rms[y_slice, x_slice]
            fit_bad_pix = bad_pix[y_slice, x_slice].copy()
            print(
                "Restricting pysersic fit to the central cutout:"
                f" half-size={fit_half_size:.1f} pixels,"
                f" cutout shape={fit_img.shape},"
                f" origin=({fit_origin_x}, {fit_origin_y}),"
                f" unmasked fit pixels={(~fit_bad_pix).sum()}"
            )
        intrinsic_model_img, conv_model_img, fit_params = run_pysersic_doublesersic(
            fit_img,
            fit_rms,
            fit_bad_pix,
            psf_img,
            inner_q_bounds=axis_ratio_bounds,
            inner_q_init=axis_ratio_init,
            outer_q_bounds=(0.2, 1.0),
            outer_q_init=0.8,
            max_train=pysersic_max_train,
            patience=pysersic_patience,
            num_round=pysersic_num_round,
            center_guess=(0.5 * (fit_img.shape[1] - 1), 0.5 * (fit_img.shape[0] - 1)),
            render_shape=gal_img.shape,
            render_origin=(fit_origin_x, fit_origin_y),
        )
        resid = gal_img - conv_model_img
        deconv_img = intrinsic_model_img + resid
        deconv_img[bad_pix] = np.nan
        extra_header = {
            "PYPROF": ("doublesersic", "pysersic profile type"),
            "PYNROUND": (pysersic_num_round, "pysersic MAP learning-rate rounds"),
            "PYMTRN": (pysersic_max_train, "pysersic MAP max steps per round"),
            "PYPAT": (pysersic_patience, "pysersic MAP early-stop patience"),
        }
        header_map = {
            "flux": "PYFLUX",
            "f_1": "PYF1",
            "r_eff_1": "PYRE1",
            "r_eff_2": "PYRE2",
            "ellip_1": "PYE1",
            "ellip_2": "PYE2",
            "theta": "PYTHETA",
            "n_1": "PYN1",
            "n_2": "PYN2",
            "n": "PYN",
            "xc": "PYXC",
            "yc": "PYYC",
        }
        for key, hdr_key in header_map.items():
            if key in fit_params:
                extra_header[hdr_key] = (float(np.asarray(fit_params[key])), key)
        write_deconvolved_image(
            gal_img_path,
            method,
            deconv_img,
            psf_fwhm_est,
            extra_header=extra_header,
        )
        return

    raise ValueError(f"Unknown deconvolution method: {method}")


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
    parser.add_argument(
        "--deconv-method",
        choices=["imcascade", "wiener", "richardson_lucy", "pysersic"],
        required=True,
        help="Single deconvolution method to apply.",
    )
    parser.add_argument(
        "--rl-num-iter",
        type=int,
        default=4,
        help="Number of Richardson-Lucy iterations when that method is used.",
    )
    parser.add_argument(
        "--pysersic-max-train",
        type=int,
        default=500,
        help="Maximum pysersic MAP optimizer steps per round.",
    )
    parser.add_argument(
        "--pysersic-patience",
        type=int,
        default=75,
        help="Pysersic MAP early-stopping patience per round.",
    )
    parser.add_argument(
        "--pysersic-num-round",
        type=int,
        default=2,
        help="Number of pysersic MAP learning-rate rounds.",
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
    deconv_method = args.deconv_method
    rl_num_iter = args.rl_num_iter
    pysersic_max_train = args.pysersic_max_train
    pysersic_patience = args.pysersic_patience
    pysersic_num_round = args.pysersic_num_round
    print(
        f"Deconvolving the stack image in z={z1}-{z2} and m={m1}-{m2} and q={q1}-{q2} in {filter} band..."
    )
    print(
        f"Selected deconvolution method: {deconv_method}"
        f" (RL iterations={rl_num_iter},"
        f" pysersic max_train={pysersic_max_train},"
        f" patience={pysersic_patience}, rounds={pysersic_num_round})"
    )

    avg_z = 0.5 * (z1 + z2)
    angular_diameter_distance = cosmo.angular_diameter_distance(avg_z)
    pixel_scale_rad = (pixel_scale * u.arcsec).to(u.rad)
    pixel_to_mpc = pixel_scale_rad.value * angular_diameter_distance
    pixel_to_kpc = pixel_to_mpc.to(u.kpc).value
    if m1 >= 9.0 and m2 <= 10.0:
        bound_pix_rad = 100 / pixel_to_kpc
    if m1 >= 10.0 and m2 <= 11.0:
        bound_pix_rad = 300 / pixel_to_kpc
    if m1 >= 11.0 and m2 <= 12.0:
        bound_pix_rad = 600 / pixel_to_kpc

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
            method=deconv_method,
            axis_ratio_init=0.5 * (q1 + q2),
            axis_ratio_bounds=(q1, q2),
            pixel_scale=pixel_scale,
            bound_pix_rad=bound_pix_rad,
            rl_num_iter=rl_num_iter,
            pysersic_max_train=pysersic_max_train,
            pysersic_patience=pysersic_patience,
            pysersic_num_round=pysersic_num_round,
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
            method=deconv_method,
            axis_ratio_init=0.5 * (q1 + q2),
            axis_ratio_bounds=(q1, q2),
            pixel_scale=pixel_scale,
            bound_pix_rad=bound_pix_rad,
            rl_num_iter=rl_num_iter,
            pysersic_max_train=pysersic_max_train,
            pysersic_patience=pysersic_patience,
            pysersic_num_round=pysersic_num_round,
        )
