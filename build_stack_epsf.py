#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=hmemq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --job-name=epsf_stack
#SBATCH --output=/gpfs01/home/ppzhg/logs/ero_psf/%j.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/ero_psf/%j.err
# fmt: on
"""Replay the galaxy rotation and median stack on the coadd ePSF library."""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits
from skimage.transform import rotate

from psf_pipeline import (
    DEFAULT_OUTPUT_ROOT,
    FILTER_CONFIGS,
    OUTPUT_PIXEL_SCALE,
    base_fits_header,
    coadd_dir,
    coadd_filename,
    coadd_phase_axis,
    coadd_phase_records,
    epsf_wcs,
    file_sha256,
    final_dir,
    normalize_filter,
    rotation_angles,
    write_json,
)

try:
    import bottleneck as bn

    _nanmedian = bn.nanmedian
except ImportError:
    _nanmedian = np.nanmedian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate every member of the 10x10 coadd phase library at ten fixed "
            "angles, then median-combine the 1000 realizations."
        )
    )
    parser.add_argument("filter", choices=FILTER_CONFIGS, help="I, Y, J, or H")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"PSF product root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument(
        "--median-stripe-width",
        type=int,
        default=8,
        help="Output columns median-combined at once (default: 8)",
    )
    parser.add_argument(
        "--keep-rotated-cube",
        action="store_true",
        help="Keep the float32 1000-image scratch cube after a successful run",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rotate_like_stack_z(
    image: np.ndarray,
    angle_deg: float,
    weight_thresh: float = 0.5,
) -> np.ndarray:
    """Apply the exact NaN-aware rotation settings used by stack_z.py."""
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    values = np.where(finite, image, 0.0).astype(np.float32, copy=False)
    weights = finite.astype(np.float32, copy=False)
    values_rotated = rotate(
        values,
        angle_deg,
        resize=False,
        order=1,
        mode="constant",
        cval=0.0,
        clip=False,
        preserve_range=True,
    )
    weights_rotated = rotate(
        weights,
        angle_deg,
        resize=False,
        order=1,
        mode="constant",
        cval=0.0,
        clip=False,
        preserve_range=True,
    )
    values_rotated = np.asarray(values_rotated, dtype=np.float32)
    weights_rotated = np.asarray(weights_rotated, dtype=np.float32)
    output = np.full_like(values_rotated, np.nan)
    valid = weights_rotated > np.float32(weight_thresh)
    np.divide(values_rotated, weights_rotated, out=output, where=valid)
    return output


def _rotate_phase_to_cube(job):
    phase_path, first_output, angles, cube_path, cube_shape = job
    image = np.asarray(fits.getdata(phase_path), dtype=np.float32)
    if image.shape != tuple(cube_shape[1:]):
        raise ValueError(f"Shape mismatch for {phase_path}: {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Non-finite input pixels in {phase_path}")

    cube = np.memmap(cube_path, dtype=np.float32, mode="r+", shape=cube_shape)
    sums = []
    for offset, angle in enumerate(angles):
        rotated = rotate_like_stack_z(image, float(angle))
        cube[first_output + offset] = rotated
        sums.append(float(np.nansum(rotated, dtype=np.float64)))
    cube.flush()
    del cube
    return first_output, sums


def median_cube(cube: np.memmap, stripe_width: int) -> np.ndarray:
    _, height, width = cube.shape
    output = np.empty((height, width), dtype=np.float32)
    for x0 in range(0, width, stripe_width):
        x1 = min(x0 + stripe_width, width)
        # stack_z.py converts its working stack to float64 before taking the
        # median. The narrow stripe keeps that behavior without a large RAM peak.
        block = np.asarray(cube[:, :, x0:x1], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            output[:, x0:x1] = _nanmedian(block, axis=0).astype(np.float32)
        print(f"Median combined columns {x0}:{x1}/{width}")
    return output


def main() -> None:
    args = parse_args()
    filter_name = normalize_filter(args.filter)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.median_stripe_width < 1:
        raise ValueError("--median-stripe-width must be positive")

    all_phase_records = coadd_phase_records(include_center=True)
    phase_directory = coadd_dir(args.output_root, filter_name)
    all_phase_paths = [
        phase_directory / coadd_filename(filter_name, record)
        for record in all_phase_records
    ]
    missing = [path for path in all_phase_paths if not path.exists()]
    if missing:
        example = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} coadd phase ePSFs; examples:\n{example}"
        )

    phase_manifest_records = []
    for record, path in zip(all_phase_records, all_phase_paths, strict=True):
        header = fits.getheader(path)
        if header.get("PSFTYPE") != "COADD_EPSF":
            raise ValueError(f"Unexpected PSFTYPE in {path}")
        if header.get("FILTER") != filter_name:
            raise ValueError(f"Unexpected filter in {path}")
        if header.get("RESAMP") != "LANCZOS2":
            raise ValueError(f"Unexpected resampler in {path}")
        if not np.isclose(header.get("PHASEX"), record.phase_x, atol=1.0e-9):
            raise ValueError(f"Unexpected PHASEX in {path}")
        if not np.isclose(header.get("PHASEY"), record.phase_y, atol=1.0e-9):
            raise ValueError(f"Unexpected PHASEY in {path}")
        phase_manifest_records.append(
            {
                "index": record.index,
                "phase_x": record.phase_x,
                "phase_y": record.phase_y,
                "is_center_byproduct": record.is_center,
                "path": str(path),
                "raw_sum": float(header["RAWSUM"]),
            }
        )

    phase_manifest_path = phase_directory / "manifest.json"
    write_json(
        phase_manifest_path,
        {
            "filter": filter_name,
            "output_pixel_scale_arcsec": OUTPUT_PIXEL_SCALE,
            "science_phase_axis": coadd_phase_axis().tolist(),
            "science_phase_count": 100,
            "center_byproduct_index": 100,
            "swarp_resampling_type": "LANCZOS2",
            "swarp_combine_type": "AVERAGE",
            "records": phase_manifest_records,
        },
    )
    phase_manifest_hash = file_sha256(phase_manifest_path)

    phase_paths = all_phase_paths[:-1]

    image_shape = fits.getdata(phase_paths[0], memmap=True).shape
    if len(image_shape) != 2 or image_shape[0] != image_shape[1]:
        raise ValueError(f"Expected a square 2D ePSF, got {image_shape}")
    angles = rotation_angles()
    sample_count = len(phase_paths) * len(angles)
    cube_shape = (sample_count, *image_shape)

    output_directory = final_dir(args.output_root, filter_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"epsf_stack_{filter_name}_0p3.fits"
    metadata_path = output_path.with_suffix(".json")
    scratch_directory = output_directory / ".work"
    scratch_directory.mkdir(parents=True, exist_ok=True)
    cube_path = scratch_directory / "rotated_epsf_cube_float32.dat"

    if output_path.exists() and not args.overwrite:
        existing_header = fits.getheader(output_path)
        if existing_header.get("COADDSHA") != phase_manifest_hash:
            raise ValueError(
                f"Coadd phase library changed for {output_path}; rerun with --overwrite"
            )
        print(f"Keeping existing {output_path}")
        return
    if cube_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Scratch cube {cube_path} already exists; inspect it or rerun with --overwrite"
            )
        cube_path.unlink()

    cube_bytes = (
        int(np.prod(cube_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    )
    with cube_path.open("wb") as stream:
        stream.truncate(cube_bytes)
    print(
        f"Allocated {cube_path} with shape {cube_shape} "
        f"({cube_bytes / 1024**3:.1f} GiB)"
    )

    succeeded = False
    try:
        jobs = [
            (
                path,
                phase_index * len(angles),
                angles,
                cube_path,
                cube_shape,
            )
            for phase_index, path in enumerate(phase_paths)
        ]
        worker_count = min(args.workers, len(jobs))
        rotation_sums = np.empty(sample_count, dtype=float)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for first_output, sums in executor.map(_rotate_phase_to_cube, jobs):
                rotation_sums[first_output : first_output + len(sums)] = sums
                print(
                    f"Rotated phase {first_output // len(angles) + 1}/"
                    f"{len(phase_paths)}"
                )

        cube = np.memmap(cube_path, dtype=np.float32, mode="r", shape=cube_shape)
        median_image = median_cube(cube, args.median_stripe_width)
        del cube
        if not np.all(np.isfinite(median_image)):
            raise ValueError("The final median ePSF contains non-finite pixels")
        raw_sum = float(np.sum(median_image, dtype=np.float64))
        if not np.isfinite(raw_sum) or raw_sum <= 0:
            raise ValueError(f"Invalid final median sum {raw_sum}")
        median_image /= raw_sum

        npix = image_shape[0]
        profile_radius = float(fits.getheader(phase_paths[0])["PRMAX"])
        header = epsf_wcs(npix, OUTPUT_PIXEL_SCALE, 0.0, 0.0).to_header()
        header.extend(
            base_fits_header(
                filter_name=filter_name,
                psf_type="STACK_EPSF",
                pixel_scale=OUTPUT_PIXEL_SCALE,
                phase_x=0.0,
                phase_y=0.0,
                profile_radius_arcsec=profile_radius,
            ),
            update=True,
        )
        header["NPHASE"] = (len(phase_paths), "coadd phase ePSFs")
        header["COADDSHA"] = phase_manifest_hash
        header["NANGLE"] = (len(angles), "rotation angles per phase")
        header["NSTACK"] = (sample_count, "images in pixelwise median")
        header["COMBTYPE"] = "MEDIAN"
        header["ROTFUNC"] = "skimage"
        header["ROTORDER"] = (1, "bilinear interpolation as in stack_z.py")
        header["ROTRESIZ"] = (False, "skimage rotate resize setting")
        header["WTHRESH"] = (0.5, "rotated finite-pixel weight threshold")
        header["RAWSUM"] = (raw_sum, "median sum before unit normalization")
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        fits.writeto(temporary, median_image, header=header, overwrite=True)
        os.replace(temporary, output_path)

        write_json(
            metadata_path,
            {
                "filter": filter_name,
                "output_path": str(output_path),
                "output_pixel_scale_arcsec": OUTPUT_PIXEL_SCALE,
                "output_shape": list(image_shape),
                "coadd_phase_axis": coadd_phase_axis().tolist(),
                "coadd_phase_count": len(phase_paths),
                "coadd_manifest": str(phase_manifest_path),
                "coadd_manifest_sha256": phase_manifest_hash,
                "center_byproduct_included": False,
                "rotation_angles_deg": angles.tolist(),
                "rotation_function": "stack_z.py NaN-aware skimage.transform.rotate",
                "rotation_order": 1,
                "rotation_resize": False,
                "rotation_weight_threshold": 0.5,
                "stack_combine": "pixelwise nanmedian",
                "stack_count": sample_count,
                "raw_median_sum": raw_sum,
                "rotated_sum_min": float(np.min(rotation_sums)),
                "rotated_sum_max": float(np.max(rotation_sums)),
                "rotated_cube_path": str(cube_path) if args.keep_rotated_cube else None,
                "rotated_cube_dtype": "float32",
                "rotated_cube_shape": list(cube_shape),
                "phase_inputs": [str(path) for path in phase_paths],
            },
        )
        succeeded = True
        print(f"Wrote final effective ePSF to {output_path}")
    finally:
        if succeeded and not args.keep_rotated_cube:
            cube_path.unlink(missing_ok=True)
            print(f"Deleted disposable rotation cube {cube_path}")
        elif cube_path.exists():
            print(f"Retained rotation cube {cube_path}")


if __name__ == "__main__":
    main()
