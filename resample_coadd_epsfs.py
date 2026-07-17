#!/gpfs01/home/ppzhg/.conda/envs/icl-py313/bin/python3
# fmt: off
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --job-name=epsf_coadd
#SBATCH --output=/gpfs01/home/ppzhg/logs/ero_psf/%A_%a.out
#SBATCH --error=/gpfs01/home/ppzhg/logs/ero_psf/%A_%a.err
# fmt: on
"""Build one phase-conditioned 0.3 arcsec coadd ePSF with SWarp."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from psf_pipeline import (
    DEFAULT_OUTPUT_ROOT,
    FILTER_CONFIGS,
    OUTPUT_PIXEL_SCALE,
    REFERENCE_DEC_DEG,
    REFERENCE_RA_DEG,
    base_fits_header,
    coadd_dir,
    coadd_filename,
    coadd_phase_axis,
    coadd_phase_records,
    file_sha256,
    native_dir,
    native_phase_axis,
    normalize_filter,
    stamp_npix,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mean-combine the 36 native-phase ePSFs onto one fixed 0.3 arcsec "
            "output phase using the SWarp Lanczos2 resampler."
        )
    )
    parser.add_argument("filter", choices=FILTER_CONFIGS, help="I, Y, J, or H")
    parser.add_argument(
        "--phase-index",
        type=int,
        help="0-99 for the 10x10 midpoint grid; 100 for the center byproduct",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all 101 phases sequentially instead of one array task",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"PSF product root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--swarp",
        default="swarp",
        help="SWarp executable name or path (default: swarp)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_records(args: argparse.Namespace):
    records = coadd_phase_records(include_center=True)
    if args.all and args.phase_index is not None:
        raise ValueError("Use either --all or --phase-index, not both")
    if args.all:
        return records

    phase_index = args.phase_index
    if phase_index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        phase_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if phase_index is None:
        raise ValueError(
            "Specify --phase-index, submit as a Slurm array, or explicitly use --all"
        )
    if phase_index < 0 or phase_index >= len(records):
        raise ValueError(f"phase index must be in [0, {len(records) - 1}]")
    return [records[phase_index]]


def load_native_manifest(output_root: Path, filter_name: str) -> dict:
    manifest_path = native_dir(output_root, filter_name) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run render_native_epsfs.py first"
        )
    with manifest_path.open(encoding="ascii") as stream:
        manifest = json.load(stream)
    if manifest["filter"] != filter_name:
        raise ValueError(f"Filter mismatch in {manifest_path}")
    if len(manifest["records"]) != 36:
        raise ValueError(f"Expected 36 native phases in {manifest_path}")
    if not np.allclose(manifest["phase_axis"], native_phase_axis(), atol=1.0e-12):
        raise ValueError(f"Unexpected native phase grid in {manifest_path}")
    expected_scale = FILTER_CONFIGS[filter_name].native_pixel_scale
    if not np.isclose(
        manifest["native_pixel_scale_arcsec"], expected_scale, atol=1.0e-12
    ):
        raise ValueError(f"Unexpected native pixel scale in {manifest_path}")
    for item in manifest["records"]:
        if not Path(item["path"]).exists():
            raise FileNotFoundError(item["path"])
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def swarp_command(
    *,
    executable: str,
    native_paths: list[Path],
    image_output: Path,
    weight_output: Path,
    resample_dir: Path,
    output_npix: int,
    phase_x: float,
    phase_y: float,
    threads: int,
) -> list[str]:
    # Moving the output tangent point while the input source remains at the
    # fixed reference sky coordinate places it at the requested coadd phase.
    center_ra = REFERENCE_RA_DEG + phase_x * OUTPUT_PIXEL_SCALE / 3600.0
    center_dec = REFERENCE_DEC_DEG - phase_y * OUTPUT_PIXEL_SCALE / 3600.0
    return [
        executable,
        *[str(path) for path in native_paths],
        "-IMAGEOUT_NAME",
        str(image_output),
        "-WEIGHTOUT_NAME",
        str(weight_output),
        "-WEIGHT_TYPE",
        "NONE",
        "-RESCALE_WEIGHTS",
        "N",
        "-COMBINE",
        "Y",
        "-COMBINE_TYPE",
        "AVERAGE",
        "-CELESTIAL_TYPE",
        "EQUATORIAL",
        "-PROJECTION_TYPE",
        "TAN",
        "-CENTER_TYPE",
        "MANUAL",
        "-CENTER",
        f"{center_ra:.12f},{center_dec:.12f}",
        "-PIXELSCALE_TYPE",
        "MANUAL",
        "-PIXEL_SCALE",
        f"{OUTPUT_PIXEL_SCALE:.12f}",
        "-IMAGE_SIZE",
        f"{output_npix},{output_npix}",
        "-RESAMPLE",
        "Y",
        "-RESAMPLE_DIR",
        str(resample_dir),
        "-RESAMPLING_TYPE",
        "LANCZOS2",
        "-OVERSAMPLING",
        "0",
        "-INTERPOLATE",
        "N",
        "-FSCALASTRO_TYPE",
        "FIXED",
        "-FSCALE_DEFAULT",
        "1.0",
        "-SUBTRACT_BACK",
        "N",
        "-DELETE_TMPFILES",
        "Y",
        "-WRITE_FILEINFO",
        "N",
        "-WRITE_XML",
        "N",
        "-VERBOSE_TYPE",
        "QUIET",
        "-NTHREADS",
        str(max(1, threads)),
    ]


def validate_output_phase(
    header: fits.Header,
    output_npix: int,
    phase_x: float,
    phase_y: float,
) -> tuple[float, float, float]:
    x, y = WCS(header).world_to_pixel_values(REFERENCE_RA_DEG, REFERENCE_DEC_DEG)
    x, y = float(np.ravel(x)[0]), float(np.ravel(y)[0])
    expected_x = output_npix // 2 + phase_x
    expected_y = output_npix // 2 + phase_y
    error = float(np.hypot(x - expected_x, y - expected_y))
    if error > 1.0e-3:
        raise RuntimeError(
            "SWarp placed the source at "
            f"({x:.6f}, {y:.6f}), expected ({expected_x:.6f}, {expected_y:.6f})"
        )
    return x, y, error


def build_phase(
    *,
    filter_name: str,
    record,
    manifest: dict,
    output_root: Path,
    swarp_executable: str,
    threads: int,
    overwrite: bool,
) -> None:
    output_directory = coadd_dir(output_root, filter_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / coadd_filename(filter_name, record)
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and not overwrite:
        header = fits.getheader(output_path)
        if header.get("PSFTYPE") != "COADD_EPSF":
            raise ValueError(f"Unexpected PSFTYPE in existing {output_path}")
        if not np.isclose(header.get("PHASEX"), record.phase_x, atol=1.0e-9):
            raise ValueError(f"Unexpected PHASEX in existing {output_path}")
        if not np.isclose(header.get("PHASEY"), record.phase_y, atol=1.0e-9):
            raise ValueError(f"Unexpected PHASEY in existing {output_path}")
        if header.get("RESAMP") != "LANCZOS2":
            raise ValueError(f"Unexpected resampler in existing {output_path}")
        if header.get("NATVSHA") != manifest["manifest_sha256"]:
            raise ValueError(
                f"Native library changed for existing {output_path}; rerun with --overwrite"
            )
        print(f"Keeping existing {output_path}")
        return

    profile_radius = float(manifest["profile_radius_arcsec"])
    output_npix = stamp_npix(profile_radius, OUTPUT_PIXEL_SCALE)
    native_paths = [Path(item["path"]) for item in manifest["records"]]
    work_root = output_directory / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix=f"phase_{record.index:03d}_", dir=work_root)
    )
    resample_dir = work_dir / "resampled"
    resample_dir.mkdir()
    swarp_output = work_dir / "coadd.fits"
    weight_output = work_dir / "coadd.weight.fits"
    command = swarp_command(
        executable=swarp_executable,
        native_paths=native_paths,
        image_output=swarp_output,
        weight_output=weight_output,
        resample_dir=resample_dir,
        output_npix=output_npix,
        phase_x=record.phase_x,
        phase_y=record.phase_y,
        threads=threads,
    )

    succeeded = False
    try:
        print(
            f"Building phase {record.index:03d}: "
            f"({record.phase_x:+.3f}, {record.phase_y:+.3f})"
        )
        subprocess.run(command, check=True, cwd=work_dir)
        header = fits.getheader(swarp_output)
        landed_x, landed_y, phase_error = validate_output_phase(
            header,
            output_npix,
            record.phase_x,
            record.phase_y,
        )
        image = np.asarray(fits.getdata(swarp_output), dtype=np.float32)
        if image.shape != (output_npix, output_npix):
            raise ValueError(f"Unexpected SWarp output shape {image.shape}")
        if not np.all(np.isfinite(image)):
            raise ValueError("SWarp output contains non-finite pixels")
        raw_sum = float(np.sum(image, dtype=np.float64))
        if not np.isfinite(raw_sum) or raw_sum <= 0:
            raise ValueError(f"Invalid SWarp output sum {raw_sum}")
        image /= raw_sum

        header.extend(
            base_fits_header(
                filter_name=filter_name,
                psf_type="COADD_EPSF",
                pixel_scale=OUTPUT_PIXEL_SCALE,
                phase_x=record.phase_x,
                phase_y=record.phase_y,
                profile_radius_arcsec=profile_radius,
            ),
            update=True,
        )
        header["PHGRID"] = (
            "center" if record.is_center else "10x10 midpoint",
            "coadd phase sampling",
        )
        header["PHINDEX"] = record.index
        header["NATIVEP"] = (len(native_paths), "native phase realizations")
        header["NATVSHA"] = manifest["manifest_sha256"]
        header["COMBTYPE"] = "AVERAGE"
        header["RESAMP"] = "LANCZOS2"
        header["RAWSUM"] = (raw_sum, "sum before unit normalization")
        header["PHERR"] = (phase_error, "output phase error, pixels")
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        fits.writeto(temporary, image, header=header, overwrite=True)
        os.replace(temporary, output_path)

        write_json(
            metadata_path,
            {
                "filter": filter_name,
                "phase_index": record.index,
                "phase_x": record.phase_x,
                "phase_y": record.phase_y,
                "is_center_byproduct": record.is_center,
                "native_manifest": str(
                    native_dir(output_root, filter_name) / "manifest.json"
                ),
                "native_phase_count": len(native_paths),
                "native_manifest_sha256": manifest["manifest_sha256"],
                "output_path": str(output_path),
                "output_pixel_scale_arcsec": OUTPUT_PIXEL_SCALE,
                "output_npix": output_npix,
                "raw_sum": raw_sum,
                "landed_pixel_x": landed_x,
                "landed_pixel_y": landed_y,
                "phase_error_pixels": phase_error,
                "swarp_resampling_type": "LANCZOS2",
                "swarp_combine_type": "AVERAGE",
                "swarp_command": command,
            },
        )
        succeeded = True
        print(f"Wrote {output_path}")
    finally:
        if succeeded:
            shutil.rmtree(work_dir)
        else:
            print(f"Retained failed-task files in {work_dir}")


def main() -> None:
    args = parse_args()
    filter_name = normalize_filter(args.filter)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    swarp_executable = shutil.which(args.swarp)
    if swarp_executable is None:
        raise FileNotFoundError(f"Could not find SWarp executable {args.swarp!r}")
    manifest = load_native_manifest(args.output_root, filter_name)
    print(f"Using SWarp executable {swarp_executable}")
    print(f"Coadd phase axis: {coadd_phase_axis().tolist()}")

    for record in selected_records(args):
        build_phase(
            filter_name=filter_name,
            record=record,
            manifest=manifest,
            output_root=args.output_root,
            swarp_executable=swarp_executable,
            threads=args.threads,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
