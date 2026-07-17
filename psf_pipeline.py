"""Shared definitions for the empirical ePSF propagation workflow."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


DEFAULT_PROFILE_DIR = Path("~/euclid_data/NICL_PSFs").expanduser()
DEFAULT_OUTPUT_ROOT = Path("~/euclid_data/ero_psf").expanduser()
OUTPUT_PIXEL_SCALE = 0.3
PROFILE_PADDING_ARCSEC = 1.2
REFERENCE_RA_DEG = 180.0
REFERENCE_DEC_DEG = 0.0


@dataclass(frozen=True)
class FilterConfig:
    profile_label: str
    native_pixel_scale: float


FILTER_CONFIGS = {
    "I": FilterConfig(profile_label="VIS", native_pixel_scale=0.1),
    "Y": FilterConfig(profile_label="Y", native_pixel_scale=0.3),
    "J": FilterConfig(profile_label="J", native_pixel_scale=0.3),
    "H": FilterConfig(profile_label="H", native_pixel_scale=0.3),
}


@dataclass(frozen=True)
class PhaseRecord:
    index: int
    phase_x: float
    phase_y: float
    is_center: bool = False


def normalize_filter(filter_name: str) -> str:
    filter_name = filter_name.upper()
    if filter_name not in FILTER_CONFIGS:
        choices = ", ".join(FILTER_CONFIGS)
        raise ValueError(f"Unknown filter {filter_name!r}; choose from {choices}")
    return filter_name


def native_phase_axis() -> np.ndarray:
    """Six midpoint samples over one native detector pixel."""
    return -0.5 + (np.arange(6, dtype=float) + 0.5) / 6.0


def coadd_phase_axis() -> np.ndarray:
    """Ten midpoint samples over one 0.3 arcsec coadd pixel."""
    return -0.5 + (np.arange(10, dtype=float) + 0.5) / 10.0


def native_phase_records() -> list[PhaseRecord]:
    axis = native_phase_axis()
    return [
        PhaseRecord(index=iy * len(axis) + ix, phase_x=x, phase_y=y)
        for iy, y in enumerate(axis)
        for ix, x in enumerate(axis)
    ]


def coadd_phase_records(*, include_center: bool = True) -> list[PhaseRecord]:
    axis = coadd_phase_axis()
    records = [
        PhaseRecord(index=iy * len(axis) + ix, phase_x=x, phase_y=y)
        for iy, y in enumerate(axis)
        for ix, x in enumerate(axis)
    ]
    if include_center:
        records.append(
            PhaseRecord(index=len(records), phase_x=0.0, phase_y=0.0, is_center=True)
        )
    return records


def rotation_angles() -> np.ndarray:
    """Ten uniformly spaced major-axis rotations over [0, 180) degrees."""
    return np.arange(10, dtype=float) * 18.0


def phase_tag(value: float, decimals: int = 6) -> str:
    sign = "p" if value >= 0 else "m"
    number = f"{abs(value):.{decimals}f}".replace(".", "p")
    return f"{sign}{number}"


def filter_root(output_root: Path, filter_name: str) -> Path:
    return output_root.expanduser() / normalize_filter(filter_name)


def native_dir(output_root: Path, filter_name: str) -> Path:
    return filter_root(output_root, filter_name) / "native_phase"


def coadd_dir(output_root: Path, filter_name: str) -> Path:
    return filter_root(output_root, filter_name) / "coadd_phase"


def final_dir(output_root: Path, filter_name: str) -> Path:
    return filter_root(output_root, filter_name) / "final"


def native_filename(filter_name: str, record: PhaseRecord) -> str:
    return (
        f"epsf_native_{normalize_filter(filter_name)}_{record.index:02d}_"
        f"x{phase_tag(record.phase_x)}_y{phase_tag(record.phase_y)}.fits"
    )


def coadd_filename(filter_name: str, record: PhaseRecord) -> str:
    filter_name = normalize_filter(filter_name)
    if record.is_center:
        return f"epsf_coadd_{filter_name}_center.fits"
    return (
        f"epsf_coadd_{filter_name}_{record.index:03d}_"
        f"x{phase_tag(record.phase_x, 3)}_y{phase_tag(record.phase_y, 3)}.fits"
    )


def profile_path(profile_dir: Path, filter_name: str) -> Path:
    config = FILTER_CONFIGS[normalize_filter(filter_name)]
    return profile_dir.expanduser() / f"stitched_profs_R_SB_{config.profile_label}.npy"


def stamp_npix(
    profile_radius_arcsec: float,
    pixel_scale: float,
    padding_arcsec: float = PROFILE_PADDING_ARCSEC,
) -> int:
    half_size = int(np.ceil((profile_radius_arcsec + padding_arcsec) / pixel_scale))
    return 2 * half_size + 1


def load_radial_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    profile = np.asarray(np.load(path), dtype=float)
    if profile.ndim != 2 or profile.shape[0] != 2:
        raise ValueError(
            f"Expected a 2xN radial profile in {path}, got {profile.shape}"
        )

    radius, surface_brightness = profile
    valid = np.isfinite(radius) & np.isfinite(surface_brightness)
    radius = radius[valid]
    surface_brightness = surface_brightness[valid]
    order = np.argsort(radius)
    radius = radius[order]
    surface_brightness = surface_brightness[order]

    if radius.size < 2 or radius[0] != 0:
        raise ValueError(f"The profile in {path} must start at radius zero")
    if np.any(np.diff(radius) <= 0):
        raise ValueError(f"The profile radii in {path} must be strictly increasing")

    # Only the relative profile matters. This avoids carrying an arbitrary
    # photometric zeropoint through very large intermediate arrays.
    intensity = 10.0 ** (-0.4 * (surface_brightness - np.nanmin(surface_brightness)))
    return radius, intensity


def render_native_epsf(
    radius: np.ndarray,
    intensity: np.ndarray,
    *,
    npix: int,
    pixel_scale: float,
    phase_x: float,
    phase_y: float,
    tile_rows: int = 128,
) -> np.ndarray:
    """Sample the empirical ePSF at native pixel centers without reintegration."""
    if npix % 2 != 1:
        raise ValueError("Native ePSF stamps must have an odd number of pixels")

    center = npix // 2
    x = (np.arange(npix, dtype=float) - center - phase_x) * pixel_scale
    image = np.empty((npix, npix), dtype=np.float32)

    for y0 in range(0, npix, tile_rows):
        y1 = min(y0 + tile_rows, npix)
        y = (np.arange(y0, y1, dtype=float) - center - phase_y) * pixel_scale
        radial_distance = np.hypot(y[:, None], x[None, :])
        tile = np.interp(
            radial_distance.ravel(),
            radius,
            intensity,
            left=float(intensity[0]),
            right=0.0,
        ).reshape(radial_distance.shape)
        # The profile is surface brightness. Multiplication by native pixel
        # area puts every stamp on a common relative flux-per-pixel convention.
        image[y0:y1] = (tile * pixel_scale**2).astype(np.float32)

    return image


def epsf_wcs(
    npix: int,
    pixel_scale: float,
    phase_x: float,
    phase_y: float,
) -> WCS:
    center = npix // 2
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [center + phase_x + 1.0, center + phase_y + 1.0]
    wcs.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
    wcs.wcs.crval = [REFERENCE_RA_DEG, REFERENCE_DEC_DEG]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


def base_fits_header(
    *,
    filter_name: str,
    psf_type: str,
    pixel_scale: float,
    phase_x: float,
    phase_y: float,
    profile_radius_arcsec: float,
) -> fits.Header:
    config = FILTER_CONFIGS[normalize_filter(filter_name)]
    header = fits.Header()
    header["PSFTYPE"] = psf_type
    header["FILTER"] = normalize_filter(filter_name)
    header["PROFLAB"] = config.profile_label
    header["PIXSCALE"] = (pixel_scale, "arcsec per pixel")
    header["PHASEX"] = (phase_x, "source x phase in pixels")
    header["PHASEY"] = (phase_y, "source y phase in pixels")
    header["PRMAX"] = (profile_radius_arcsec, "profile truncation radius, arcsec")
    header["PADARC"] = (PROFILE_PADDING_ARCSEC, "zero padding beyond profile, arcsec")
    header["BUNIT"] = "relative flux"
    header["FLXSCALE"] = 1.0
    return header


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
