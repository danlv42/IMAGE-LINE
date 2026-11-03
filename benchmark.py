"""Compare two IMAGE-LINE codec implementations at an identical byte budget.

Example:
  python benchmark.py photo.png --baseline-codec ../IMAGE-LINE-main/codec.py --max-kb 200
  python benchmark.py ./samples --baseline-codec ../old/codec.py --max-kb 200 --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import codec as candidate_codec


SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_codec(path: Path):
    spec = importlib.util.spec_from_file_location("baseline_codec_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load codec module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_metrics(reference_path: Path, restored_path: Path) -> dict:
    ref = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.float64)
    got = np.asarray(Image.open(restored_path).convert("RGB"), dtype=np.float64)
    if got.shape != ref.shape:
        got = np.asarray(
            Image.fromarray(np.uint8(np.clip(got, 0, 255))).resize(
                (ref.shape[1], ref.shape[0]), Image.Resampling.LANCZOS
            ),
            dtype=np.float64,
        )

    diff = ref - got
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)

    # Global SSIM: standard SSIM formula over each RGB channel's full image.
    # This deterministic metric needs only NumPy (no scikit-image dependency).
    mu_x = np.mean(ref, axis=(0, 1))
    mu_y = np.mean(got, axis=(0, 1))
    var_x = np.mean((ref - mu_x) ** 2, axis=(0, 1))
    var_y = np.mean((got - mu_y) ** 2, axis=(0, 1))
    cov_xy = np.mean((ref - mu_x) * (got - mu_y), axis=(0, 1))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_channels = (
        (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    )
    ssim = float(np.mean(ssim_channels))
    return {
        "psnr_db": round(psnr, 4),
        "global_ssim": round(ssim, 6),
        "rgb_mae": round(float(np.mean(np.abs(diff))), 4),
    }


def run_one(module, source: Path, tag: str, max_bytes: int, temp: Path) -> dict:
    encoded = temp / f"{tag}.lsc"
    restored = temp / f"{tag}_restored.png"
    result = module.encode(str(source), str(encoded), max_bytes=max_bytes)
    module.decode(str(encoded), str(restored))
    metrics = image_metrics(source, restored)
    return {
        "codec": tag,
        "bytes": encoded.stat().st_size,
        "limit_bytes": max_bytes,
        **metrics,
        "scale": result.get("scale", ""),
        "quality": result.get("quality", ""),
        "encoded_path": str(encoded),
        "restored_path": str(restored),
    }


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported image type: {path.suffix}")
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One image or a directory of images")
    parser.add_argument("--baseline-codec", type=Path, required=True, help="Path to the previous codec.py")
    parser.add_argument("--max-kb", type=int, required=True, help="Shared output budget in KiB")
    parser.add_argument("--csv", type=Path, default=Path("benchmark_results.csv"))
    args = parser.parse_args()

    if args.max_kb <= 0:
        parser.error("--max-kb must be positive")
    images = collect_images(args.input)
    if not images:
        parser.error("No supported images found")

    baseline = load_codec(args.baseline_codec)
    budget = args.max_kb * 1024
    rows = []
    for source in images:
        with tempfile.TemporaryDirectory(prefix="image-line-bench-") as directory:
            temp = Path(directory)
            baseline_result = run_one(baseline, source, "baseline_lsc2", budget, temp)
            candidate_result = run_one(candidate_codec, source, "candidate_lsc3", budget, temp)
            rows.append({
                "image": str(source),
                "source_bytes": source.stat().st_size,
                "budget_bytes": budget,
                "baseline_bytes": baseline_result["bytes"],
                "baseline_psnr_db": baseline_result["psnr_db"],
                "baseline_global_ssim": baseline_result["global_ssim"],
                "baseline_rgb_mae": baseline_result["rgb_mae"],
                "candidate_bytes": candidate_result["bytes"],
                "candidate_psnr_db": candidate_result["psnr_db"],
                "candidate_global_ssim": candidate_result["global_ssim"],
                "candidate_rgb_mae": candidate_result["rgb_mae"],
                "psnr_delta_db": round(candidate_result["psnr_db"] - baseline_result["psnr_db"], 4),
                "global_ssim_delta": round(candidate_result["global_ssim"] - baseline_result["global_ssim"], 6),
            })
            print(
                f"{source.name}: LSC2 {baseline_result['bytes']} B, "
                f"{baseline_result['psnr_db']} dB / SSIM {baseline_result['global_ssim']} | "
                f"LSC3 {candidate_result['bytes']} B, "
                f"{candidate_result['psnr_db']} dB / SSIM {candidate_result['global_ssim']}"
            )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
