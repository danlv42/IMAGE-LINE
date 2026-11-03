"""
IMAGE-LINE LSC v3: adaptive block quantization with cached DCT transforms.
Pipeline: RGB -> YCbCr -> 4:2:0 chroma subsampling -> 8x8 DCT ->
custom quantization -> zlib-compressed coefficient arrays.
This is an educational custom format, not JPEG/WebP/AVIF.
Dependencies: Pillow, NumPy.
"""
from __future__ import annotations
import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

MAGIC = b"LSC3"
LEGACY_MAGIC = b"LSC2"
VERSION = 3
HEADER_LEN = struct.Struct(">I")
BLOCK = 8

# JPEG-inspired starting tables are used only as quantization weights;
# the container, search strategy, coefficient storage and decoder are ours.
QY = np.array([
    [16,11,10,16,24,40,51,61], [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56], [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77], [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101], [72,92,95,98,112,100,103,99]
], dtype=np.float32)
QC = np.array([
    [17,18,24,47,99,99,99,99], [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99], [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99], [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99], [99,99,99,99,99,99,99,99]
], dtype=np.float32)
ZIGZAG = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(1,6),(2,5),(3,4),(4,3),(5,2),
    (6,1),(7,0),(7,1),(6,2),(5,3),(4,4),(3,5),(2,6),
    (1,7),(2,7),(3,6),(4,5),(5,4),(6,3),(7,2),(7,3),
    (6,4),(5,5),(4,6),(3,7),(4,7),(5,6),(6,5),(7,4),
    (7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]
ZIGZAG_FLAT = np.array([r * 8 + c for r, c in ZIGZAG], dtype=np.intp)
# Keep the original 62-entry scan so files created by the first buggy MVP remain readable.
LEGACY_ZIGZAG_FLAT = ZIGZAG_FLAT.copy()
# Complete 64-position zigzag scan for new LSC2 files.
_full_zigzag = []
for _diagonal in range(15):
    _lo, _hi = max(0, _diagonal - 7), min(7, _diagonal)
    _rows = range(_hi, _lo - 1, -1) if _diagonal % 2 == 0 else range(_lo, _hi + 1)
    for _row in _rows:
        _full_zigzag.append((_row, _diagonal - _row))
ZIGZAG = _full_zigzag
ZIGZAG_FLAT = np.array([r * 8 + c for r, c in ZIGZAG], dtype=np.intp)
# Orthonormal DCT-II basis.
_C = np.empty((8, 8), dtype=np.float32)
for _u in range(8):
    _alpha = (1 / np.sqrt(8)) if _u == 0 else np.sqrt(2 / 8)
    for _x in range(8):
        _C[_u, _x] = _alpha * np.cos((2 * _x + 1) * _u * np.pi / 16)


def _quant_table(base: np.ndarray, quality: int) -> np.ndarray:
    quality = max(1, min(100, int(quality)))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.maximum(1, np.floor((base * scale + 50) / 100)).astype(np.float32)


def _blocks(channel: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = channel.shape
    ph = (BLOCK - h % BLOCK) % BLOCK
    pw = (BLOCK - w % BLOCK) % BLOCK
    padded = np.pad(channel, ((0, ph), (0, pw)), mode="edge")
    hp, wp = padded.shape
    blocks = padded.reshape(hp // 8, 8, wp // 8, 8).transpose(0, 2, 1, 3)
    return blocks.reshape(-1, 8, 8), hp, wp


def _dct_coefficients(channel: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Transform one channel once; keep coefficients in zigzag order for reuse."""
    blocks, hp, wp = _blocks(channel.astype(np.float32) - 128.0)
    coeff = np.einsum("ij,bjk,lk->bil", _C, blocks, _C, optimize=True)
    zigzag_coeff = coeff.reshape(-1, 64)[:, ZIGZAG_FLAT]
    return zigzag_coeff.astype(np.float32), hp, wp


def _quantize_coefficients(coeff: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Quantize cached zigzag DCT coefficients without repeating the DCT."""
    table_flat = table.reshape(64)[ZIGZAG_FLAT]
    return np.rint(coeff / table_flat[None, :]).astype(">i2")


def _dct_quantize(channel: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, int, int]:
    # Compatibility helper retained for internal callers and tests.
    coeff, hp, wp = _dct_coefficients(channel)
    return _quantize_coefficients(coeff, table), hp, wp


def _idct_dequantize(stream: np.ndarray, hp: int, wp: int, table: np.ndarray) -> np.ndarray:
    stream = np.asarray(stream)
    if stream.ndim != 2:
        raise ValueError("Coefficient stream must be a 2D block array.")
    if stream.shape[1] == 62:
        # Backward compatibility for .lsc files created by the initial MVP,
        # whose zigzag table accidentally omitted two coefficients per block.
        q_flat = np.zeros((stream.shape[0], 64), dtype=np.float32)
        q_flat[:, LEGACY_ZIGZAG_FLAT] = stream.astype(np.float32)
    elif stream.shape[1] == 64:
        q_flat = stream.astype(np.float32)[:, np.argsort(ZIGZAG_FLAT)]
    else:
        raise ValueError(f"Invalid coefficient count per block: {stream.shape[1]}")
    q = q_flat.reshape(-1, 8, 8)
    coeff = q * table[None, :, :]
    blocks = np.einsum("ji,bjk,kl->bil", _C, coeff, _C, optimize=True) + 128.0
    rows = blocks.reshape(hp // 8, wp // 8, 8, 8).transpose(0, 2, 1, 3)
    return np.clip(rows.reshape(hp, wp), 0, 255)


def _idct_dequantize_adaptive(stream: np.ndarray, hp: int, wp: int,
                                 table: np.ndarray, factor_ids: np.ndarray) -> np.ndarray:
    """Inverse transform with stored per-block quantization multiplier ids."""
    stream = np.asarray(stream)
    if stream.ndim != 2 or stream.shape[1] not in (62, 64):
        raise ValueError("Invalid adaptive coefficient stream.")
    if factor_ids.size != stream.shape[0] or np.any(factor_ids > 2):
        raise ValueError("Invalid adaptive quantization factors.")
    if stream.shape[1] == 62:
        q_flat = np.zeros((stream.shape[0], 64), dtype=np.float32)
        q_flat[:, LEGACY_ZIGZAG_FLAT] = stream.astype(np.float32)
    else:
        q_flat = stream.astype(np.float32)[:, np.argsort(ZIGZAG_FLAT)]
    q = q_flat.reshape(-1, 8, 8)
    multipliers = np.array([0.90, 1.00, 1.10], dtype=np.float32)[factor_ids]
    coeff = q * table[None, :, :] * multipliers[:, None, None]
    blocks = np.einsum("ji,bjk,kl->bil", _C, coeff, _C, optimize=True) + 128.0
    rows = blocks.reshape(hp // 8, wp // 8, 8, 8).transpose(0, 2, 1, 3)
    return np.clip(rows.reshape(hp, wp), 0, 255)


def _prepare_scale(image: Image.Image, scale: float) -> dict:
    """Resize, subsample and transform channels once for all quality trials."""
    w = max(1, round(image.width * scale))
    h = max(1, round(image.height * scale))
    scaled = image.resize((w, h), Image.Resampling.LANCZOS).convert("YCbCr")
    y, cb, cr = [np.asarray(ch, dtype=np.uint8) for ch in scaled.split()]

    # 4:2:0 chroma subsampling: vision is less sensitive to fine color detail.
    cw, ch = max(1, (w + 1) // 2), max(1, (h + 1) // 2)
    cb_small = np.asarray(
        Image.fromarray(cb).resize((cw, ch), Image.Resampling.BOX), dtype=np.uint8
    )
    cr_small = np.asarray(
        Image.fromarray(cr).resize((cw, ch), Image.Resampling.BOX), dtype=np.uint8
    )

    y_coeff, yhp, ywp = _dct_coefficients(y)
    cb_coeff, chp, cwp = _dct_coefficients(cb_small)
    cr_coeff, _, _ = _dct_coefficients(cr_small)
    return {
        "width": w, "height": h,
        "y_coeff": y_coeff, "cb_coeff": cb_coeff, "cr_coeff": cr_coeff,
        "y_padded": [yhp, ywp], "c_padded": [chp, cwp],
        "original_width": image.width, "original_height": image.height,
        "scale": float(scale),
    }


def _activity_factors(coeff: np.ndarray) -> np.ndarray:
    """Choose gentle per-block quantization multipliers from AC activity ranks.

    Low-activity blocks receive slightly finer quantization; highly textured
    blocks receive slightly coarser quantization. Only three levels are used,
    so the side information is one byte per block and easy to decode.
    """
    if coeff.shape[0] == 0:
        return np.empty((0,), dtype=np.uint8)
    activity = np.mean(np.abs(coeff[:, 1:]), axis=1)
    low, high = np.quantile(activity, [0.30, 0.70])
    # 0 -> 0.90x, 1 -> 1.00x, 2 -> 1.10x base quantization.
    factors = np.ones(activity.shape, dtype=np.uint8)
    factors[activity <= low] = 0
    factors[activity >= high] = 2
    return factors


def _quantize_adaptive(coeff: np.ndarray, table: np.ndarray,
                       factor_ids: np.ndarray) -> np.ndarray:
    """Quantize zigzag DCT coefficients with one multiplier per block."""
    table_flat = table.reshape(64)[ZIGZAG_FLAT]
    multipliers = np.array([0.90, 1.00, 1.10], dtype=np.float32)[factor_ids]
    adjusted = table_flat[None, :] * multipliers[:, None]
    return np.rint(coeff / adjusted).astype(">i2")


def _pack_prepared(prepared: dict, quality: int) -> bytes:
    """Quantize and entropy-compress one prepared scale at a selected quality."""
    ty, tc = _quant_table(QY, quality), _quant_table(QC, quality)
    y_factors = _activity_factors(prepared["y_coeff"])
    cb_factors = _activity_factors(prepared["cb_coeff"])
    cr_factors = _activity_factors(prepared["cr_coeff"])
    yq = _quantize_adaptive(prepared["y_coeff"], ty, y_factors)
    cbq = _quantize_adaptive(prepared["cb_coeff"], tc, cb_factors)
    crq = _quantize_adaptive(prepared["cr_coeff"], tc, cr_factors)

    # Store one compact factor id per block after the int16 coefficient arrays.
    raw = (
        yq.tobytes() + cbq.tobytes() + crq.tobytes()
        + y_factors.tobytes() + cb_factors.tobytes() + cr_factors.tobytes()
    )
    compressed = zlib.compress(raw, level=6)
    header = {
        "version": VERSION, "width": prepared["width"], "height": prepared["height"],
        "original_width": prepared["original_width"],
        "original_height": prepared["original_height"],
        "quality": int(quality), "scale": prepared["scale"],
        "y_padded": prepared["y_padded"], "c_padded": prepared["c_padded"],
        "y_coeff_count": int(yq.size), "c_coeff_count": int(cbq.size),
        "y_factor_count": int(y_factors.size),
        "c_factor_count": int(cb_factors.size),
        "adaptive_quantization": "activity-rank-v1",
        "chroma": "4:2:0", "transform": "8x8 DCT-II",
        "quantization": "custom scalar tables with block activity multipliers",
        "payload": "zlib int16 zigzag + uint8 block factors"
    }
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + HEADER_LEN.pack(len(hb)) + hb + compressed


def _encode_candidate(image: Image.Image, scale: float, quality: int) -> bytes:
    # Publicly useful convenience wrapper; encode() reuses prepared transforms.
    return _pack_prepared(_prepare_scale(image, scale), quality)


def encode(input_path: str, output_path: str, max_bytes: int) -> dict:
    if max_bytes <= 0:
        raise ValueError("The size limit must be positive.")
    source = Path(input_path)
    image = Image.open(source).convert("RGB")
    # Coarse-to-fine parameter grid: substantially fewer expensive DCT passes
    # than the original 0.025 scale / 5-quality exhaustive grid.
    scales = [round(float(v), 3) for v in np.arange(1.0, 0.049, -0.05)]
    qualities = list(range(100, 0, -10)) + [1]
    best = None
    smallest_size = None
    for scale in scales:
        # DCT is the expensive stage. Cache transformed channels per scale instead
        # of recomputing them for every quality trial.
        prepared = _prepare_scale(image, scale)
        for quality in qualities:
            data = _pack_prepared(prepared, quality)
            size = len(data)
            smallest_size = size if smallest_size is None else min(smallest_size, size)
            if size <= max_bytes:
                # Preserve the existing ranking for format compatibility; later
                # versions can replace this heuristic with a perceptual objective.
                score = (prepared["width"] * prepared["height"], quality, -size)
                if best is None or score > best[0]:
                    best = (score, data, scale, quality)
    if best is None:
        raise ValueError(f"Could not fit into {max_bytes} bytes. Smallest tested candidate: {smallest_size} bytes.")
    Path(output_path).write_bytes(best[1])
    return {
        "input": str(source), "output": str(output_path),
        "input_bytes": source.stat().st_size, "output_bytes": len(best[1]),
        "limit_bytes": max_bytes, "width": image.width, "height": image.height,
        "scale": best[2], "quality": best[3],
        "compression_ratio": round(source.stat().st_size / len(best[1]), 2)
    }


def decode(input_path: str, output_path: str) -> dict:
    data = Path(input_path).read_bytes()
    if len(data) < 8 or data[:4] not in (MAGIC, LEGACY_MAGIC):
        raise ValueError("Not an LSC file (expected magic LSC2 or LSC3).")
    header_size = HEADER_LEN.unpack(data[4:8])[0]
    header_end = 8 + header_size
    if header_end > len(data):
        raise ValueError("Corrupt LSC header.")
    header = json.loads(data[8:header_end].decode("utf-8"))
    file_version = int(header.get("version", 0))
    if (data[:4] == MAGIC and file_version != VERSION) or (
        data[:4] == LEGACY_MAGIC and file_version != 2
    ):
        raise ValueError("Unsupported LSC version.")
    raw = zlib.decompress(data[header_end:])
    y_count, c_count = int(header["y_coeff_count"]), int(header["c_coeff_count"])
    coefficient_bytes = (y_count + 2 * c_count) * 2
    adaptive = header.get("adaptive_quantization") == "activity-rank-v1"
    y_factor_count = int(header.get("y_factor_count", 0)) if adaptive else 0
    c_factor_count = int(header.get("c_factor_count", 0)) if adaptive else 0
    expected_bytes = coefficient_bytes + y_factor_count + 2 * c_factor_count
    if len(raw) != expected_bytes:
        raise ValueError("Coefficient/factor data size does not match the header.")
    coeff = np.frombuffer(raw[:coefficient_bytes], dtype=">i2")
    if adaptive:
        factor_data = np.frombuffer(raw[coefficient_bytes:], dtype=np.uint8)
        y_factors = factor_data[:y_factor_count]
        cb_factors = factor_data[y_factor_count:y_factor_count + c_factor_count]
        cr_factors = factor_data[y_factor_count + c_factor_count:]
    else:
        y_factors = cb_factors = cr_factors = None
    quality = int(header["quality"])
    ty, tc = _quant_table(QY, quality), _quant_table(QC, quality)
    yhp, ywp = map(int, header["y_padded"])
    chp, cwp = map(int, header["c_padded"])
    y_blocks = (yhp // 8) * (ywp // 8)
    c_blocks = (chp // 8) * (cwp // 8)
    if y_count == y_blocks * 64 and c_count == c_blocks * 64:
        coefficients_per_block = 64
    elif y_count == y_blocks * 62 and c_count == c_blocks * 62:
        coefficients_per_block = 62
    else:
        raise ValueError("Coefficient counts do not match padded image dimensions.")
    yq = coeff[:y_count].reshape(y_blocks, coefficients_per_block)
    cbq = coeff[y_count:y_count + c_count].reshape(c_blocks, coefficients_per_block)
    crq = coeff[y_count + c_count:].reshape(c_blocks, coefficients_per_block)
    if adaptive:
        y = _idct_dequantize_adaptive(yq, yhp, ywp, ty, y_factors)
        cb = _idct_dequantize_adaptive(cbq, chp, cwp, tc, cb_factors)
        cr = _idct_dequantize_adaptive(crq, chp, cwp, tc, cr_factors)
    else:
        y = _idct_dequantize(yq, yhp, ywp, ty)
        cb = _idct_dequantize(cbq, chp, cwp, tc)
        cr = _idct_dequantize(crq, chp, cwp, tc)
    y = y[:int(header["height"]), :int(header["width"])]
    cw, ch = max(1, (int(header["width"]) + 1) // 2), max(1, (int(header["height"]) + 1) // 2)
    cb = cb[:ch, :cw]
    cr = cr[:ch, :cw]
    cb_img = Image.fromarray(np.uint8(np.clip(cb, 0, 255))).resize((int(header["width"]), int(header["height"])), Image.Resampling.BILINEAR)
    cr_img = Image.fromarray(np.uint8(np.clip(cr, 0, 255))).resize((int(header["width"]), int(header["height"])), Image.Resampling.BILINEAR)
    y_img = Image.fromarray(np.uint8(np.clip(y, 0, 255)))
    restored = Image.merge("YCbCr", (y_img, cb_img, cr_img)).convert("RGB")
    original_size = (int(header["original_width"]), int(header["original_height"]))
    if restored.size != original_size:
        restored = restored.resize(original_size, Image.Resampling.LANCZOS)
    restored.save(output_path)
    return {"input": input_path, "output": output_path, "size": restored.size, "quality": quality}


def main():
    parser = argparse.ArgumentParser(description="IMAGE-LINE custom LSC v3 lossy codec")
    sub = parser.add_subparsers(dest="command", required=True)
    enc = sub.add_parser("encode", help="Encode an image into .lsc")
    enc.add_argument("input")
    enc.add_argument("output")
    enc.add_argument("--max-kb", type=int, required=True)
    dec = sub.add_parser("decode", help="Decode .lsc into a standard image")
    dec.add_argument("input")
    dec.add_argument("output")
    args = parser.parse_args()
    try:
        result = encode(args.input, args.output, args.max_kb * 1024) if args.command == "encode" else decode(args.input, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
