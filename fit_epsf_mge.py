#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
"""Fit a wide-angle stacked ePSF with pixel-integrated circular Gaussians.

The fit keeps separate constraints on central pixels, discrete core apertures,
the logarithmic wing profile, and wide-angle tail flux. Outputs include the
``sigma_pix amplitude`` table consumed by the deconvolution workflow, a model
FITS image, fit metadata, and a diagnostic plot.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares, nnls
from scipy.special import erf

from psf_pipeline import DEFAULT_OUTPUT_ROOT, FILTER_CONFIGS, file_sha256, write_json


CORE_LOG_WEIGHT = 2.0
CORE_PIXEL_WEIGHT = 8.0
CORE_APERTURE_WEIGHT = 16.0
WING_WEIGHT = 4.0
TAIL_WEIGHT = 2.0


@dataclass(frozen=True)
class FitTargets:
    core_x_pix: np.ndarray
    core_y_pix: np.ndarray
    core_flux: np.ndarray
    core_multiplicity: np.ndarray
    core_aperture_radius_pix: np.ndarray
    core_aperture_flux: np.ndarray
    wing_inner_arcsec: np.ndarray
    wing_outer_arcsec: np.ndarray
    wing_surface_brightness: np.ndarray
    tail_radius_arcsec: np.ndarray
    tail_flux: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", choices=FILTER_CONFIGS, help="I, Y, J, or H")
    parser.add_argument(
        "--input",
        type=Path,
        help="Final stacked ePSF FITS file (default: pipeline product for filter)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"PSF product root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--components", type=int, default=24)
    parser.add_argument("--sigma-min-arcsec", type=float, default=0.03)
    parser.add_argument("--sigma-max-arcsec", type=float, default=300.0)
    parser.add_argument("--core-radius-arcsec", type=float, default=3.0)
    parser.add_argument("--wing-bins", type=int, default=120)
    parser.add_argument("--tail-points", type=int, default=40)
    parser.add_argument("--prune-amplitude", type=float, default=1.0e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pixel_integral_design(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    sigma_arcsec: np.ndarray,
    pixel_scale: float,
) -> np.ndarray:
    """Unit-flux Gaussian fractions in square pixels at integer offsets."""
    sigma_pix = sigma_arcsec / pixel_scale
    root_two_sigma = np.sqrt(2.0) * sigma_pix[None, :]
    x = x_pix[:, None]
    y = y_pix[:, None]
    x_fraction = 0.5 * (
        erf((x + 0.5) / root_two_sigma) - erf((x - 0.5) / root_two_sigma)
    )
    y_fraction = 0.5 * (
        erf((y + 0.5) / root_two_sigma) - erf((y - 0.5) / root_two_sigma)
    )
    return x_fraction * y_fraction


def annular_design(
    inner_arcsec: np.ndarray,
    outer_arcsec: np.ndarray,
    sigma_arcsec: np.ndarray,
) -> np.ndarray:
    """Mean surface brightness of unit-flux Gaussians in circular annuli."""
    sigma2 = sigma_arcsec[None, :] ** 2
    inner2 = inner_arcsec[:, None] ** 2
    outer2 = outer_arcsec[:, None] ** 2
    annular_flux = np.exp(-inner2 / (2.0 * sigma2)) - np.exp(-outer2 / (2.0 * sigma2))
    area = np.pi * (outer2 - inner2)
    return annular_flux / area


def tail_design(radius_arcsec: np.ndarray, sigma_arcsec: np.ndarray) -> np.ndarray:
    """Flux outside each circular radius for unit-flux Gaussians."""
    return np.exp(-(radius_arcsec[:, None] ** 2) / (2.0 * sigma_arcsec[None, :] ** 2))


def grouped_core_targets(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    pixel_scale: float,
    core_radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """D4-average central pixels to match a circular Gaussian model."""
    center_x_int = int(round(center_x))
    center_y_int = int(round(center_y))
    if not np.isclose(center_x, center_x_int, atol=1.0e-5) or not np.isclose(
        center_y, center_y_int, atol=1.0e-5
    ):
        raise ValueError("The final stacked ePSF must be centered on a pixel")

    radius_pix = int(np.ceil(core_radius_arcsec / pixel_scale))
    grouped: dict[tuple[int, int], list[float]] = {}
    for dy in range(-radius_pix, radius_pix + 1):
        for dx in range(-radius_pix, radius_pix + 1):
            if np.hypot(dx, dy) * pixel_scale > core_radius_arcsec:
                continue
            key = tuple(sorted((abs(dx), abs(dy))))
            grouped.setdefault(key, []).append(
                float(image[center_y_int + dy, center_x_int + dx])
            )

    keys = sorted(grouped, key=lambda value: (np.hypot(*value), *value))
    core_x = np.asarray([key[0] for key in keys], dtype=float)
    core_y = np.asarray([key[1] for key in keys], dtype=float)
    core_flux = np.asarray([np.mean(grouped[key]) for key in keys], dtype=float)
    core_multiplicity = np.asarray([len(grouped[key]) for key in keys], dtype=float)
    valid = np.isfinite(core_flux) & (core_flux > 0)
    return (
        core_x[valid],
        core_y[valid],
        core_flux[valid],
        core_multiplicity[valid],
    )


def cumulative_flux_at_radii(
    radius_image: np.ndarray, image: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    order = np.argsort(radius_image, axis=None)
    sorted_radius = radius_image.ravel()[order]
    cumulative = np.cumsum(image.ravel()[order], dtype=np.float64)
    indices = np.searchsorted(sorted_radius, radii, side="right") - 1
    clipped = np.clip(indices, 0, len(cumulative) - 1)
    enclosed = np.where(indices >= 0, cumulative[clipped], 0.0)
    return enclosed


def build_targets(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    pixel_scale: float,
    core_radius_arcsec: float,
    fit_radius_arcsec: float,
    wing_bins: int,
    tail_points: int,
) -> FitTargets:
    core_x, core_y, core_flux, core_multiplicity = grouped_core_targets(
        image, center_x, center_y, pixel_scale, core_radius_arcsec
    )
    core_radius_pix = np.hypot(core_x, core_y)
    core_aperture_radius_pix = np.unique(core_radius_pix)
    core_aperture_flux = np.asarray(
        [
            np.sum(
                core_flux[core_radius_pix <= radius + 1.0e-12]
                * core_multiplicity[core_radius_pix <= radius + 1.0e-12]
            )
            for radius in core_aperture_radius_pix
        ]
    )

    y, x = np.indices(image.shape, dtype=np.float32)
    radius_image = np.hypot(x - center_x, y - center_y) * pixel_scale
    edges = np.geomspace(core_radius_arcsec, fit_radius_arcsec, wing_bins + 1)
    flat_radius = radius_image.ravel()
    flat_image = image.ravel()
    selected = (
        np.isfinite(flat_image) & (flat_radius >= edges[0]) & (flat_radius < edges[-1])
    )
    bins = np.searchsorted(edges, flat_radius[selected], side="right") - 1
    count = np.bincount(bins, minlength=wing_bins)
    flux_sum = np.bincount(bins, weights=flat_image[selected], minlength=wing_bins)
    wing_surface_brightness = np.divide(
        flux_sum,
        count * pixel_scale**2,
        out=np.full_like(flux_sum, np.nan),
        where=count > 0,
    )
    valid_wing = (
        (count > 0)
        & np.isfinite(wing_surface_brightness)
        & (wing_surface_brightness > 0)
    )

    tail_radius = np.geomspace(core_radius_arcsec, 0.9 * fit_radius_arcsec, tail_points)
    enclosed = cumulative_flux_at_radii(radius_image, image, tail_radius)
    tail_flux = np.sum(image, dtype=np.float64) - enclosed
    valid_tail = np.isfinite(tail_flux) & (tail_flux > 0)

    return FitTargets(
        core_x_pix=core_x,
        core_y_pix=core_y,
        core_flux=core_flux,
        core_multiplicity=core_multiplicity,
        core_aperture_radius_pix=core_aperture_radius_pix,
        core_aperture_flux=core_aperture_flux,
        wing_inner_arcsec=edges[:-1][valid_wing],
        wing_outer_arcsec=edges[1:][valid_wing],
        wing_surface_brightness=wing_surface_brightness[valid_wing],
        tail_radius_arcsec=tail_radius[valid_tail],
        tail_flux=tail_flux[valid_tail],
    )


def build_designs(
    targets: FitTargets, sigma_arcsec: np.ndarray, pixel_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    core_design = pixel_integral_design(
        targets.core_x_pix,
        targets.core_y_pix,
        sigma_arcsec,
        pixel_scale,
    )
    core_radius = np.hypot(targets.core_x_pix, targets.core_y_pix)
    aperture_membership = (
        core_radius[None, :] <= targets.core_aperture_radius_pix[:, None] + 1.0e-12
    )
    core_aperture_design = (
        aperture_membership * targets.core_multiplicity[None, :]
    ) @ core_design
    return (
        core_design,
        core_aperture_design,
        annular_design(
            targets.wing_inner_arcsec,
            targets.wing_outer_arcsec,
            sigma_arcsec,
        ),
        tail_design(targets.tail_radius_arcsec, sigma_arcsec),
    )


def initial_amplitudes(
    targets: FitTargets, sigma_arcsec: np.ndarray, pixel_scale: float
) -> np.ndarray:
    core_design, core_aperture_design, wing_design, tail_matrix = build_designs(
        targets, sigma_arcsec, pixel_scale
    )
    groups = (
        (core_design, targets.core_flux, CORE_LOG_WEIGHT),
        (
            core_aperture_design,
            targets.core_aperture_flux,
            CORE_APERTURE_WEIGHT,
        ),
        (wing_design, targets.wing_surface_brightness, WING_WEIGHT),
        (tail_matrix, targets.tail_flux, TAIL_WEIGHT),
    )
    weighted_design = []
    weighted_target = []
    for design, target, group_weight in groups:
        scale = np.sqrt(group_weight / len(target)) / target
        weighted_design.append(design * scale[:, None])
        weighted_target.append(target * scale)

    # Log residuals preserve the dynamic range; this term preserves the flux
    # distribution among the high-signal central pixels.
    core_norm = np.sqrt(np.sum(targets.core_multiplicity * targets.core_flux**2))
    core_scale = np.sqrt(CORE_PIXEL_WEIGHT * targets.core_multiplicity) / core_norm
    weighted_design.append(core_design * core_scale[:, None])
    weighted_target.append(targets.core_flux * core_scale)
    weighted_design.append(np.ones((1, len(sigma_arcsec))) * 100.0)
    weighted_target.append(np.asarray([100.0]))
    amplitude, _ = nnls(
        np.vstack(weighted_design), np.concatenate(weighted_target), maxiter=10000
    )
    if np.sum(amplitude) <= 0:
        raise RuntimeError("Could not initialize positive Gaussian amplitudes")
    return amplitude / np.sum(amplitude)


def amplitudes_from_ratios(
    log_ratios: np.ndarray, component_count: int, reference_index: int
) -> np.ndarray:
    logits = np.zeros(component_count, dtype=float)
    free = np.arange(component_count) != reference_index
    logits[free] = log_ratios
    logits -= np.max(logits)
    amplitude = np.exp(logits)
    return amplitude / np.sum(amplitude)


def fit_ordered_mge(
    targets: FitTargets,
    *,
    pixel_scale: float,
    sigma_initial: np.ndarray,
    amplitude_initial: np.ndarray,
    sigma_lower: np.ndarray,
    sigma_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, object]:
    component_count = len(sigma_initial)
    reference_index = int(np.argmax(amplitude_initial))
    free = np.arange(component_count) != reference_index
    safe_amplitude = np.maximum(amplitude_initial, 1.0e-14)
    log_ratios = np.log(safe_amplitude[free] / safe_amplitude[reference_index])
    initial = np.concatenate((np.log(sigma_initial), log_ratios))
    lower = np.concatenate((np.log(sigma_lower), np.full(component_count - 1, -35.0)))
    upper = np.concatenate((np.log(sigma_upper), np.full(component_count - 1, 35.0)))

    tiny = np.finfo(float).tiny

    def residual(parameters: np.ndarray) -> np.ndarray:
        sigma_arcsec = np.exp(parameters[:component_count])
        amplitude = amplitudes_from_ratios(
            parameters[component_count:], component_count, reference_index
        )
        core_design, core_aperture_design, wing_design, tail_matrix = build_designs(
            targets, sigma_arcsec, pixel_scale
        )
        core_model = np.maximum(core_design @ amplitude, tiny)
        core_aperture_model = np.maximum(core_aperture_design @ amplitude, tiny)
        wing_model = np.maximum(wing_design @ amplitude, tiny)
        tail_model = np.maximum(tail_matrix @ amplitude, tiny)
        core_norm = np.sqrt(np.sum(targets.core_multiplicity * targets.core_flux**2))
        return np.concatenate(
            (
                np.log(core_model / targets.core_flux)
                * np.sqrt(CORE_LOG_WEIGHT / len(targets.core_flux)),
                (core_model - targets.core_flux)
                * np.sqrt(CORE_PIXEL_WEIGHT * targets.core_multiplicity)
                / core_norm,
                np.log(core_aperture_model / targets.core_aperture_flux)
                * np.sqrt(CORE_APERTURE_WEIGHT / len(targets.core_aperture_flux)),
                np.log(wing_model / targets.wing_surface_brightness)
                * np.sqrt(WING_WEIGHT / len(targets.wing_surface_brightness)),
                np.log(tail_model / targets.tail_flux)
                * np.sqrt(TAIL_WEIGHT / len(targets.tail_flux)),
            )
        )

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.03,
        x_scale="jac",
        max_nfev=4000,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    if not result.success:
        raise RuntimeError(f"MGE optimization failed: {result.message}")
    sigma_arcsec = np.exp(result.x[:component_count])
    amplitude = amplitudes_from_ratios(
        result.x[component_count:], component_count, reference_index
    )
    return sigma_arcsec, amplitude, result


def render_mge(
    shape: tuple[int, int],
    sigma_arcsec: np.ndarray,
    amplitude: np.ndarray,
    pixel_scale: float,
) -> np.ndarray:
    if shape[0] % 2 != 1 or shape[1] % 2 != 1:
        raise ValueError("MGE rendering requires an odd-sized centered image")
    x = np.arange(shape[1], dtype=float) - shape[1] // 2
    y = np.arange(shape[0], dtype=float) - shape[0] // 2
    model = np.zeros(shape, dtype=np.float64)
    for sigma, weight in zip(sigma_arcsec, amplitude, strict=True):
        sigma_pix = sigma / pixel_scale
        denominator = np.sqrt(2.0) * sigma_pix
        x_fraction = 0.5 * (erf((x + 0.5) / denominator) - erf((x - 0.5) / denominator))
        y_fraction = 0.5 * (erf((y + 0.5) / denominator) - erf((y - 0.5) / denominator))
        model += weight * np.outer(y_fraction, x_fraction)
    return model


def radial_profile(
    image: np.ndarray, pixel_scale: float, fit_radius_arcsec: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_y, center_x = (np.asarray(image.shape) - 1.0) / 2.0
    y, x = np.indices(image.shape, dtype=np.float32)
    radius_image = np.hypot(x - center_x, y - center_y) * pixel_scale
    fine_limit = min(12.0, fit_radius_arcsec)
    fine_edges = np.arange(0.0, fine_limit + pixel_scale, pixel_scale)
    outer_edges = np.geomspace(fine_edges[-1], fit_radius_arcsec, 101)[1:]
    edges = np.concatenate((fine_edges, outer_edges))
    selected = radius_image.ravel() < edges[-1]
    bins = np.searchsorted(edges, radius_image.ravel()[selected], side="right") - 1
    count = np.bincount(bins, minlength=len(edges) - 1)
    flux = np.bincount(bins, weights=image.ravel()[selected], minlength=len(edges) - 1)
    valid = count > 0
    radius = 0.5 * (edges[:-1] + edges[1:])
    surface_brightness = flux / count / pixel_scale**2
    return radius[valid], surface_brightness[valid], edges


def enclosed_flux(
    image: np.ndarray, pixel_scale: float, radius_arcsec: np.ndarray
) -> np.ndarray:
    center_y, center_x = (np.asarray(image.shape) - 1.0) / 2.0
    y, x = np.indices(image.shape, dtype=np.float32)
    radius_image = np.hypot(x - center_x, y - center_y) * pixel_scale
    return cumulative_flux_at_radii(radius_image, image, radius_arcsec)


def save_diagnostic_plot(
    output_path: Path,
    *,
    data_image: np.ndarray,
    model_image: np.ndarray,
    sigma_arcsec: np.ndarray,
    amplitude: np.ndarray,
    pixel_scale: float,
    fit_radius_arcsec: float,
) -> dict[str, float]:
    radius, data_profile, _ = radial_profile(data_image, pixel_scale, fit_radius_arcsec)
    model_radius, model_profile, _ = radial_profile(
        model_image, pixel_scale, fit_radius_arcsec
    )
    if not np.allclose(radius, model_radius):
        raise RuntimeError("Data and MGE radial grids differ")
    valid = (data_profile > 0) & (model_profile > 0)
    fractional = model_profile[valid] / data_profile[valid] - 1.0
    log_residual_dex = np.log10(model_profile[valid] / data_profile[valid])

    ee_radius = np.geomspace(pixel_scale / 2.0, fit_radius_arcsec, 160)
    data_ee = enclosed_flux(data_image, pixel_scale, ee_radius)
    model_ee = enclosed_flux(model_image, pixel_scale, ee_radius)
    ee_difference = model_ee - data_ee

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.5, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.15, 1.15], "hspace": 0.07},
    )
    axes[0].loglog(
        radius[data_profile > 0],
        data_profile[data_profile > 0],
        label="Final ePSF",
        color="#343a40",
        linewidth=2.2,
    )
    axes[0].loglog(
        radius[model_profile > 0],
        model_profile[model_profile > 0],
        label="Pixel-integrated MGE",
        color="#d1495b",
        linewidth=1.8,
    )
    component_radius = np.geomspace(0.01, fit_radius_arcsec, 800)
    for sigma, weight in zip(sigma_arcsec, amplitude, strict=True):
        component = (
            weight
            / (2.0 * np.pi * sigma**2)
            * np.exp(-(component_radius**2) / (2.0 * sigma**2))
        )
        axes[0].loglog(
            component_radius,
            component,
            color="#6c757d",
            linewidth=0.65,
            alpha=0.32,
        )
    axes[0].set_ylabel(r"Unit-flux surface brightness (arcsec$^{-2}$)")
    axes[0].set_title(f"{len(sigma_arcsec)}-component ePSF Gaussian decomposition")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.22, which="both")
    plotted_profile = np.concatenate((data_profile[valid], model_profile[valid]))
    axes[0].set_ylim(0.5 * np.min(plotted_profile), 2.0 * np.max(plotted_profile))

    axes[1].semilogx(radius[valid], fractional, color="#d1495b", linewidth=1.4)
    axes[1].axhline(0.0, color="#343a40", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("MGE / ePSF - 1")
    axes[1].grid(alpha=0.22, which="both")

    axes[2].semilogx(ee_radius, ee_difference, color="#00798c", linewidth=1.5)
    axes[2].axhline(0.0, color="#343a40", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("Radius (arcsec)")
    axes[2].set_ylabel(r"EE$_{\rm MGE}$ - EE$_{\rm ePSF}$")
    axes[2].grid(alpha=0.22, which="both")
    axes[2].set_xlim(pixel_scale / 2.0, fit_radius_arcsec)

    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.png")
    fig.savefig(temporary, dpi=200, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, output_path)

    max_ee_index = int(np.argmax(np.abs(ee_difference)))
    beyond_one_arcsec = ee_radius >= 1.0
    outer_ee_index = np.flatnonzero(beyond_one_arcsec)[
        np.argmax(np.abs(ee_difference[beyond_one_arcsec]))
    ]
    return {
        "profile_rms_log10": float(np.sqrt(np.mean(log_residual_dex**2))),
        "profile_median_abs_fraction": float(np.median(np.abs(fractional))),
        "profile_p95_abs_fraction": float(np.quantile(np.abs(fractional), 0.95)),
        "max_abs_encircled_flux_error": float(np.max(np.abs(ee_difference))),
        "max_abs_encircled_flux_error_radius_arcsec": float(ee_radius[max_ee_index]),
        "max_abs_encircled_flux_error_beyond_1_arcsec": float(
            np.abs(ee_difference[outer_ee_index])
        ),
        "max_abs_encircled_flux_error_beyond_1_arcsec_radius_arcsec": float(
            ee_radius[outer_ee_index]
        ),
    }


def main() -> None:
    args = parse_args()
    filter_name = args.filter.upper()
    if args.components < 2:
        raise ValueError("At least two Gaussian components are required")
    if not 0 < args.sigma_min_arcsec < args.sigma_max_arcsec:
        raise ValueError("Require 0 < sigma-min < sigma-max")
    if args.core_radius_arcsec <= 0:
        raise ValueError("Core radius must be positive")
    if args.wing_bins < 1 or args.tail_points < 1:
        raise ValueError("Wing bins and tail points must be positive")
    if not 0 <= args.prune_amplitude < 1:
        raise ValueError("Prune amplitude must be in [0, 1)")

    input_path = args.input or (
        args.output_root.expanduser()
        / filter_name
        / "final"
        / f"epsf_stack_{filter_name}_0p3.fits"
    )
    input_path = input_path.expanduser()
    output_prefix = input_path.with_name(f"{input_path.stem}_mge")
    text_path = output_prefix.with_suffix(".txt")
    json_path = output_prefix.with_suffix(".json")
    model_path = output_prefix.with_name(f"{output_prefix.name}_model.fits")
    plot_path = output_prefix.with_name(f"{output_prefix.name}_diagnostic.png")
    outputs = (text_path, json_path, model_path, plot_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    source_hash = file_sha256(input_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        if len(existing) != len(outputs):
            raise FileExistsError(
                "Only some MGE outputs exist; inspect them or use --overwrite:\n"
                + "\n".join(str(path) for path in existing)
            )
        with json_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("input_sha256") != source_hash:
            raise ValueError(
                f"Input ePSF changed for {text_path}; rerun with --overwrite"
            )
        print(f"Keeping existing MGE products for {input_path}")
        return

    with fits.open(input_path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
        source_header = hdul[0].header.copy()
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"Expected a square 2D ePSF, got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Input ePSF contains non-finite pixels")
    image_sum = float(np.sum(image, dtype=np.float64))
    if image_sum <= 0:
        raise ValueError(f"Invalid input ePSF sum {image_sum}")
    image /= image_sum

    pixel_scale = float(source_header["PIXSCALE"])
    phase_x = float(source_header.get("PHASEX", 0.0))
    phase_y = float(source_header.get("PHASEY", 0.0))
    center_x = (image.shape[1] - 1.0) / 2.0 + phase_x
    center_y = (image.shape[0] - 1.0) / 2.0 + phase_y
    fit_radius_arcsec = float(source_header["PRMAX"])
    if fit_radius_arcsec <= args.core_radius_arcsec:
        raise ValueError("Fit radius must exceed the core radius")
    targets = build_targets(
        image,
        center_x=center_x,
        center_y=center_y,
        pixel_scale=pixel_scale,
        core_radius_arcsec=args.core_radius_arcsec,
        fit_radius_arcsec=fit_radius_arcsec,
        wing_bins=args.wing_bins,
        tail_points=args.tail_points,
    )

    sigma_edges = np.geomspace(
        args.sigma_min_arcsec,
        args.sigma_max_arcsec,
        args.components + 1,
    )
    sigma_lower = sigma_edges[:-1]
    sigma_upper = sigma_edges[1:]
    sigma_initial = np.sqrt(sigma_edges[:-1] * sigma_edges[1:])
    amplitude_initial = initial_amplitudes(targets, sigma_initial, pixel_scale)
    sigma_arcsec, amplitude, result = fit_ordered_mge(
        targets,
        pixel_scale=pixel_scale,
        sigma_initial=sigma_initial,
        amplitude_initial=amplitude_initial,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
    )

    for _ in range(args.components):
        active = amplitude >= args.prune_amplitude
        if np.all(active):
            break
        if np.count_nonzero(active) < 2:
            raise RuntimeError("Amplitude pruning left fewer than two components")
        sigma_lower = sigma_lower[active]
        sigma_upper = sigma_upper[active]
        sigma_arcsec, amplitude, result = fit_ordered_mge(
            targets,
            pixel_scale=pixel_scale,
            sigma_initial=sigma_arcsec[active],
            amplitude_initial=amplitude[active] / np.sum(amplitude[active]),
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
        )
    else:
        raise RuntimeError("Amplitude pruning did not converge")

    order = np.argsort(sigma_arcsec)
    sigma_arcsec = sigma_arcsec[order]
    amplitude = amplitude[order]
    amplitude /= np.sum(amplitude)
    sigma_pix = sigma_arcsec / pixel_scale
    model_image = render_mge(image.shape, sigma_arcsec, amplitude, pixel_scale)
    model_sum = float(np.sum(model_image, dtype=np.float64))

    model_header = source_header.copy()
    model_header["PSFTYPE"] = "MGE_MODEL"
    model_header["NCOMP"] = len(sigma_arcsec)
    model_header["SRCPSF"] = input_path.name
    model_header["PSFSHA"] = source_hash
    model_header["MODSUM"] = model_sum
    temporary_model = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    fits.writeto(
        temporary_model,
        model_image.astype(np.float32),
        header=model_header,
        overwrite=True,
    )
    os.replace(temporary_model, model_path)

    diagnostics = save_diagnostic_plot(
        plot_path,
        data_image=image,
        model_image=model_image,
        sigma_arcsec=sigma_arcsec,
        amplitude=amplitude,
        pixel_scale=pixel_scale,
        fit_radius_arcsec=fit_radius_arcsec,
    )
    diagnostics.update(
        {
            "central_pixel_data": float(
                image[image.shape[0] // 2, image.shape[1] // 2]
            ),
            "central_pixel_model": float(
                model_image[model_image.shape[0] // 2, model_image.shape[1] // 2]
            ),
            "model_stamp_sum": model_sum,
        }
    )

    temporary_text = text_path.with_name(f".{text_path.name}.{os.getpid()}.tmp")
    np.savetxt(
        temporary_text,
        np.column_stack((sigma_pix, amplitude)),
        header="sigma_pix amplitude",
        fmt="%.12e",
    )
    os.replace(temporary_text, text_path)
    write_json(
        json_path,
        {
            "filter": filter_name,
            "input_path": str(input_path),
            "input_sha256": source_hash,
            "input_pixel_scale_arcsec": pixel_scale,
            "input_shape": list(image.shape),
            "fit_radius_arcsec": fit_radius_arcsec,
            "core_radius_arcsec": args.core_radius_arcsec,
            "core_pixel_group_count": len(targets.core_flux),
            "core_aperture_constraint_count": len(targets.core_aperture_flux),
            "wing_bin_count": len(targets.wing_surface_brightness),
            "tail_constraint_count": len(targets.tail_flux),
            "requested_component_count": args.components,
            "active_component_count": len(sigma_arcsec),
            "prune_amplitude": args.prune_amplitude,
            "sigma_arcsec": sigma_arcsec.tolist(),
            "sigma_pix": sigma_pix.tolist(),
            "amplitude": amplitude.tolist(),
            "amplitude_sum": float(np.sum(amplitude)),
            "objective_weights": {
                "core_log": CORE_LOG_WEIGHT,
                "core_pixel_flux": CORE_PIXEL_WEIGHT,
                "core_aperture_flux": CORE_APERTURE_WEIGHT,
                "wing": WING_WEIGHT,
                "tail": TAIL_WEIGHT,
            },
            "optimizer": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": result.message,
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "function_evaluations": int(result.nfev),
            },
            "diagnostics": diagnostics,
            "text_output": str(text_path),
            "model_output": str(model_path),
            "plot_output": str(plot_path),
        },
    )

    print(f"Wrote {text_path}")
    print(f"Wrote {model_path}")
    print(f"Wrote {plot_path}")
    print(f"Active components: {len(sigma_arcsec)}")
    print(f"Model stamp sum: {model_sum:.10f}")
    for key, value in diagnostics.items():
        print(f"{key}: {value:.6g}")


if __name__ == "__main__":
    main()
