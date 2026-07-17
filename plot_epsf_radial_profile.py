#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
"""Compare a final stacked ePSF radial profile with its exposure model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from psf_pipeline import DEFAULT_OUTPUT_ROOT, DEFAULT_PROFILE_DIR, FILTER_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", choices=FILTER_CONFIGS, help="I, Y, J, or H")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help=f"Input profile directory (default: {DEFAULT_PROFILE_DIR})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"PSF product root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output plot path (default: beside the final ePSF)",
    )
    return parser.parse_args()


def load_unit_flux_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    radius, surface_brightness = np.asarray(np.load(path), dtype=float)
    valid = np.isfinite(radius) & np.isfinite(surface_brightness)
    radius = radius[valid]
    surface_brightness = surface_brightness[valid]
    order = np.argsort(radius)
    radius = radius[order]
    surface_brightness = surface_brightness[order]

    intensity = 10.0 ** (-0.4 * (surface_brightness - np.nanmin(surface_brightness)))
    total = 2.0 * np.pi * np.trapezoid(intensity * radius, radius)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid radial-profile integral {total}")
    return radius, intensity / total


def annular_profile(
    image: np.ndarray,
    *,
    pixel_scale: float,
    phase_x: float,
    phase_y: float,
    model_radius: np.ndarray,
    model_intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = image.shape
    center_x = (nx - 1) / 2.0 + phase_x
    center_y = (ny - 1) / 2.0 + phase_y
    y, x = np.indices(image.shape, dtype=np.float32)
    radius_image = np.hypot(x - center_x, y - center_y) * pixel_scale

    profile_limit = float(model_radius[-1])
    fine_limit = min(12.0, profile_limit)
    fine_edges = np.arange(0.0, fine_limit + pixel_scale, pixel_scale)
    outer_edges = np.geomspace(fine_edges[-1], profile_limit, 81)[1:]
    edges = np.concatenate((fine_edges, outer_edges))

    flat_radius = radius_image.ravel()
    flat_image = image.ravel()
    finite = np.isfinite(flat_image) & (flat_radius < edges[-1])
    bins = np.searchsorted(edges, flat_radius[finite], side="right") - 1
    count = np.bincount(bins, minlength=len(edges) - 1)
    image_sum = np.bincount(bins, weights=flat_image[finite], minlength=len(edges) - 1)

    # Evaluate the exposure model at the same pixel-centre radii so the ratio
    # is not biased by the variable-width outer annuli.
    model_samples = np.interp(
        flat_radius[finite], model_radius, model_intensity, right=0.0
    )
    model_sum = np.bincount(bins, weights=model_samples, minlength=len(edges) - 1)

    good = count > 0
    bin_radius = 0.5 * (edges[:-1] + edges[1:])
    final_profile = image_sum / count / pixel_scale**2
    sampled_model = model_sum / count
    return bin_radius[good], final_profile[good], sampled_model[good]


def main() -> None:
    args = parse_args()
    filter_name = args.filter.upper()
    config = FILTER_CONFIGS[filter_name]
    profile_path = (
        args.profile_dir.expanduser()
        / f"stitched_profs_R_SB_{config.profile_label}.npy"
    )
    final_path = (
        args.output_root.expanduser()
        / filter_name
        / "final"
        / f"epsf_stack_{filter_name}_0p3.fits"
    )
    output_path = args.output or final_path.with_name(
        f"epsf_stack_{filter_name}_0p3_radial_profile.png"
    )

    model_radius, model_intensity = load_unit_flux_profile(profile_path)
    with fits.open(final_path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header
    image /= np.sum(image, dtype=np.float64)

    pixel_scale = float(header["PIXSCALE"])
    radius, final_profile, sampled_model = annular_profile(
        image,
        pixel_scale=pixel_scale,
        phase_x=float(header.get("PHASEX", 0.0)),
        phase_y=float(header.get("PHASEY", 0.0)),
        model_radius=model_radius,
        model_intensity=model_intensity,
    )
    valid_ratio = (final_profile > 0) & (sampled_model > 0)

    fig, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06},
    )
    positive_model = (model_radius > 0) & (model_intensity > 0)
    axis.loglog(
        model_radius[positive_model],
        model_intensity[positive_model],
        color="#343a40",
        linewidth=2.0,
        label="Single-exposure empirical ePSF",
    )
    axis.loglog(
        radius[final_profile > 0],
        final_profile[final_profile > 0],
        color="#d1495b",
        linewidth=1.8,
        label="Final rotated-stack ePSF",
    )
    axis.set_ylabel(r"Unit-flux surface brightness (arcsec$^{-2}$)")
    axis.set_title(f"NISP {filter_name}-band ePSF radial profiles")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(alpha=0.22, which="both")

    ratio_axis.semilogx(
        radius[valid_ratio],
        final_profile[valid_ratio] / sampled_model[valid_ratio],
        color="#d1495b",
        linewidth=1.6,
    )
    ratio_axis.axhline(1.0, color="#343a40", linewidth=1.0, linestyle="--")
    ratio_axis.set_xlabel("Radius (arcsec)")
    ratio_axis.set_ylabel("Final / single")
    ratio_axis.grid(alpha=0.22, which="both")
    ratio_axis.set_xlim(pixel_scale / 2.0, model_radius[-1])

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
