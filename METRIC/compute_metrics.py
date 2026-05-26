from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
FUSED_DIR = ROOT / "DATA" / "FUSED"
REF_DIR = ROOT / "DATA" / "REF"
OUTPUT_DIR = ROOT / "DATA" / "OUTPUT"


def load_mat_array(path: Path, variable: str | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        data = loadmat(path)
        items = {
            key: value
            for key, value in data.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }
    except NotImplementedError:
        with h5py.File(path, "r") as file:
            items = {
                key: np.array(file[key]).transpose()
                for key in file.keys()
                if hasattr(file[key], "shape")
            }

    if variable:
        if variable not in items:
            raise KeyError(f"{variable!r} was not found in {path.name}")
        return np.asarray(items[variable], dtype=np.float64)

    numeric = [
        (key, value)
        for key, value in items.items()
        if np.issubdtype(value.dtype, np.number) and value.ndim in (2, 3)
    ]
    if not numeric:
        raise ValueError(f"No numeric 2-D/3-D array found in {path}")

    numeric.sort(key=lambda item: item[1].size, reverse=True)
    return np.asarray(numeric[0][1], dtype=np.float64)


def to_hwc(array: np.ndarray, reference_shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {array.shape}")

    if reference_shape and array.shape == reference_shape:
        return array

    # Common hyperspectral layout from MATLAB/Python models: (bands, rows, cols).
    if array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
        array = np.moveaxis(array, 0, -1)

    if reference_shape and array.shape != reference_shape:
        for axes in ((1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)):
            candidate = np.transpose(array, axes)
            if candidate.shape == reference_shape:
                return candidate
        raise ValueError(f"Shape mismatch: fused {array.shape}, reference {reference_shape}")

    return array


def psnr(ref: np.ndarray, fused: np.ndarray) -> tuple[float, np.ndarray]:
    mse = np.mean((ref - fused) ** 2, axis=(0, 1))
    peak = np.max(ref, axis=(0, 1))
    values = np.full(ref.shape[2], np.inf, dtype=np.float64)
    valid = mse > 0
    values[valid] = 10.0 * np.log10((peak[valid] ** 2) / mse[valid])
    return float(np.mean(values)), values


def rmse(ref: np.ndarray, fused: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ref - fused) ** 2)))


def sam(ref: np.ndarray, fused: np.ndarray) -> tuple[float, np.ndarray]:
    dot = np.sum(ref * fused, axis=2)
    norm = np.linalg.norm(ref, axis=2) * np.linalg.norm(fused, axis=2)
    valid = norm > 0
    cosines = np.ones_like(dot)
    cosines[valid] = np.clip(dot[valid] / norm[valid], -1.0, 1.0)
    angle_map = np.arccos(cosines)
    angle_deg = np.degrees(angle_map[valid]).mean() if np.any(valid) else 0.0
    return float(angle_deg), angle_map


def ergas(ref: np.ndarray, fused: np.ndarray, ratio: float) -> float:
    mse = np.mean((ref - fused) ** 2, axis=(0, 1))
    mean_ref = np.mean(ref, axis=(0, 1))
    valid = mean_ref != 0
    if not np.any(valid):
        return float("nan")
    return float((100.0 / ratio) * np.sqrt(np.mean(mse[valid] / (mean_ref[valid] ** 2))))


def ssim(ref: np.ndarray, fused: np.ndarray) -> float:
    values: list[float] = []
    for band in range(ref.shape[2]):
        ref_band = ref[:, :, band]
        fused_band = fused[:, :, band]
        data_range = float(ref_band.max() - ref_band.min())
        if data_range == 0:
            values.append(1.0 if np.allclose(ref_band, fused_band) else 0.0)
            continue
        values.append(
            float(structural_similarity(ref_band, fused_band, data_range=data_range))
        )
    return float(np.mean(values))


def correlation(ref: np.ndarray, fused: np.ndarray) -> float:
    ref_flat = ref.reshape(-1)
    fused_flat = fused.reshape(-1)
    if np.std(ref_flat) == 0 or np.std(fused_flat) == 0:
        return float("nan")
    return float(np.corrcoef(ref_flat, fused_flat)[0, 1])


def compute(ref: np.ndarray, fused: np.ndarray, ratio: float) -> dict[str, float | list[float] | str]:
    psnr_mean, psnr_all = psnr(ref, fused)
    sam_mean, _ = sam(ref, fused)
    return {
        "PSNR": psnr_mean,
        "SSIM": ssim(ref, fused),
        "SAM": sam_mean,
        "ERGAS": ergas(ref, fused, ratio),
        "RMSE": rmse(ref, fused),
        "CC": correlation(ref, fused),
        "PSNR_ALL": psnr_all.tolist(),
    }


def compute_best_orientation(
    ref: np.ndarray, fused: np.ndarray, ratio: float
) -> dict[str, float | list[float] | str]:
    candidates = [("normal", fused)]
    swapped = np.transpose(fused, (1, 0, 2))
    if swapped.shape == ref.shape:
        candidates.append(("swap_hw", swapped))

    best_name = ""
    best_metrics: dict[str, float | list[float] | str] | None = None
    for name, candidate in candidates:
        metrics = compute(ref, candidate, ratio)
        if best_metrics is None or float(metrics["PSNR"]) > float(best_metrics["PSNR"]):
            best_name = name
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("No valid orientation candidate was found")
    best_metrics["ORIENTATION"] = best_name
    return best_metrics


def save_result(
    output_dir: Path, name: str, metrics: dict[str, float | list[float] | str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    public_metrics = {key: value for key, value in metrics.items() if key != "PSNR_ALL"}
    with (output_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with (output_dir / f"{name}_metrics.txt").open("w", encoding="utf-8") as file:
        for key, value in public_metrics.items():
            if isinstance(value, str):
                file.write(f"{key}: {value}\n")
            else:
                file.write(f"{key}: {value:.8f}\n")

    with (output_dir / f"{name}_psnr_per_band.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["band", "psnr"])
        for index, value in enumerate(metrics["PSNR_ALL"], start=1):
            writer.writerow([index, value])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fused-image quality metrics.")
    parser.add_argument("--fused-dir", type=Path, default=FUSED_DIR)
    parser.add_argument("--ref-dir", type=Path, default=REF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ratio", type=float, default=12.0, help="ERGAS scale ratio.")
    parser.add_argument("--fused-var", default=None, help="Optional fused variable name.")
    parser.add_argument("--ref-var", default=None, help="Optional reference variable name.")
    args = parser.parse_args()

    ref_files = sorted(args.ref_dir.glob("*.mat"))
    fused_files = sorted(args.fused_dir.glob("*.mat"))
    if not ref_files:
        raise FileNotFoundError(f"No .mat reference file found in {args.ref_dir}")
    if not fused_files:
        raise FileNotFoundError(f"No .mat fused file found in {args.fused_dir}")

    ref = to_hwc(load_mat_array(ref_files[0], args.ref_var))
    summary_rows = []

    for fused_path in fused_files:
        fused = to_hwc(load_mat_array(fused_path, args.fused_var), ref.shape)
        metrics = compute_best_orientation(ref, fused, args.ratio)
        save_result(args.output_dir, fused_path.stem, metrics)
        summary_rows.append(
            {
                "file": fused_path.name,
                **{key: value for key, value in metrics.items() if key != "PSNR_ALL"},
            }
        )
        print(f"\n{fused_path.name}")
        for key, value in summary_rows[-1].items():
            if key != "file":
                if isinstance(value, str):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value:.8f}")

    with (args.output_dir / "metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = ["file", "ORIENTATION", "PSNR", "SSIM", "SAM", "ERGAS", "RMSE", "CC"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
