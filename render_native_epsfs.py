#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --job-name=epsf_native
#SBATCH --output=/gpfs01/home/ppzhg/logs/ero_psf/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/ero_psf/%j.err
# fmt: on
"""Render the 6x6 native-exposure phase library for one Euclid filter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from psf_pipeline import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROFILE_DIR,
    FILTER_CONFIGS,
    PROFILE_PADDING_ARCSEC,
    base_fits_header,
    epsf_wcs,
    file_sha256,
    load_radial_profile,
    native_dir,
    native_filename,
    native_phase_axis,
    native_phase_records,
    normalize_filter,
    profile_path,
    render_native_epsf,
    stamp_npix,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the empirical radial ePSF on a 6x6 native-pixel phase grid. "
            "The profile is treated as an image-domain ePSF and is not integrated "
            "over the detector pixel again."
        )
    )
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_name = normalize_filter(args.filter)
    config = FILTER_CONFIGS[filter_name]
    input_path = profile_path(args.profile_dir, filter_name)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    radius, intensity = load_radial_profile(input_path)
    profile_hash = file_sha256(input_path)
    profile_radius = float(radius[-1])
    npix = stamp_npix(profile_radius, config.native_pixel_scale)
    output_dir = native_dir(args.output_root, filter_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for record in native_phase_records():
        output_path = output_dir / native_filename(filter_name, record)
        if output_path.exists() and not args.overwrite:
            print(f"Keeping existing {output_path}")
            existing_header = fits.getheader(output_path)
            if existing_header.get("PROFSHA") != profile_hash:
                raise ValueError(
                    f"Profile changed for existing {output_path}; rerun with --overwrite"
                )
            if not np.isclose(
                existing_header.get("PHASEX"), record.phase_x, atol=1.0e-12
            ) or not np.isclose(
                existing_header.get("PHASEY"), record.phase_y, atol=1.0e-12
            ):
                raise ValueError(f"Phase mismatch in existing {output_path}")
            raw_sum = float(f"{float(existing_header['RAWSUM']):.12e}")
        else:
            print(
                f"Rendering native phase {record.index:02d}/35: "
                f"({record.phase_x:+.6f}, {record.phase_y:+.6f})"
            )
            image = render_native_epsf(
                radius,
                intensity,
                npix=npix,
                pixel_scale=config.native_pixel_scale,
                phase_x=record.phase_x,
                phase_y=record.phase_y,
            )
            raw_sum = float(f"{np.sum(image, dtype=np.float64):.12e}")
            if not np.isfinite(raw_sum) or raw_sum <= 0:
                raise ValueError(
                    f"Invalid rendered flux {raw_sum} for phase {record.index}"
                )

            header = epsf_wcs(
                npix,
                config.native_pixel_scale,
                record.phase_x,
                record.phase_y,
            ).to_header()
            header.extend(
                base_fits_header(
                    filter_name=filter_name,
                    psf_type="NATIVE_EPSF",
                    pixel_scale=config.native_pixel_scale,
                    phase_x=record.phase_x,
                    phase_y=record.phase_y,
                    profile_radius_arcsec=profile_radius,
                ),
                update=True,
            )
            header["PHGRID"] = ("6x6 midpoint", "native phase sampling")
            header["PHINDEX"] = record.index
            header["PROFSHA"] = profile_hash
            header["RAWSUM"] = (raw_sum, "sum before SWarp and final normalization")
            temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
            fits.writeto(temporary, image, header=header, overwrite=True)
            os.replace(temporary, output_path)

        manifest_records.append(
            {
                "index": record.index,
                "phase_x": record.phase_x,
                "phase_y": record.phase_y,
                "path": str(output_path),
                "raw_sum": raw_sum,
            }
        )

    manifest = {
        "filter": filter_name,
        "profile_label": config.profile_label,
        "profile_path": str(input_path),
        "profile_sha256": profile_hash,
        "profile_radius_arcsec": profile_radius,
        "profile_padding_arcsec": PROFILE_PADDING_ARCSEC,
        "native_pixel_scale_arcsec": config.native_pixel_scale,
        "stamp_npix": npix,
        "phase_axis": native_phase_axis().tolist(),
        "rendering": "radial empirical ePSF sampled at native pixel centers",
        "normalization": "common relative surface-brightness scale; not normalized per phase",
        "records": manifest_records,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote {len(manifest_records)} native phase stamps to {output_dir}")


if __name__ == "__main__":
    main()
