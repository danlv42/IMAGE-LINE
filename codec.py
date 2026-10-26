"""
LSC v1 — educational custom lossy image codec.
Format: magic + JSON header length + JSON header + zlib-compressed RGB bytes.
Requires Pillow: pip install Pillow
"""
from __future__ import annotations
import argparse
import json
import struct
import zlib
from pathlib import Path
from PIL import Image

MAGIC = b"LSC1"
VERSION = 1
HEADER_LEN = struct.Struct(">I")


def _encode_candidate(image: Image.Image, scale: float, levels: int) -> bytes:
    w = max(1, round(image.width * scale))
    h = max(1, round(image.height * scale))
    small = image.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
    if levels < 256:
        step = 255.0 / (levels - 1)
        px = small.load()
        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y]
                px[x, y] = tuple(
                    max(0, min(255, round(round(c / step) * step)))
                    for c in (r, g, b)
                )
    compressed = zlib.compress(small.tobytes(), level=9)
    header = {
        "version": VERSION, "width": w, "height": h,
        "original_width": image.width, "original_height": image.height,
        "levels": levels, "scale": scale, "pixel_format": "RGB888",
        "compression": "zlib"
    }
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + HEADER_LEN.pack(len(hb)) + hb + compressed


def encode(input_path: str, output_path: str, max_bytes: int) -> dict:
    if max_bytes <= 0:
        raise ValueError("The size limit must be positive.")
    source = Path(input_path)
    image = Image.open(source).convert("RGB")
    scales = [1.0, .90, .80, .70, .60, .50, .42, .35, .28, .22, .17, .12, .08, .05]
    levels_list = [256, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2]
    best = None
    smallest_size = None
    for scale in scales:
        for levels in levels_list:
            data = _encode_candidate(image, scale, levels)
            smallest_size = len(data) if smallest_size is None else min(smallest_size, len(data))
            if len(data) <= max_bytes:
                score = (round(image.width * scale) * round(image.height * scale), levels, -len(data))
                if best is None or score > best[0]:
                    best = (score, data, scale, levels)
    if best is None:
        raise ValueError(
            f"Could not fit into {max_bytes} bytes. Smallest tested candidate: {smallest_size} bytes."
        )
    Path(output_path).write_bytes(best[1])
    return {
        "input": str(source), "output": str(output_path),
        "input_bytes": source.stat().st_size, "output_bytes": len(best[1]),
        "limit_bytes": max_bytes, "width": image.width, "height": image.height,
        "scale": best[2], "levels": best[3],
        "compression_ratio": round(source.stat().st_size / len(best[1]), 2)
    }


def decode(input_path: str, output_path: str) -> dict:
    data = Path(input_path).read_bytes()
    if len(data) < 8 or data[:4] != MAGIC:
        raise ValueError("Not an LSC v1 file.")
    header_size = HEADER_LEN.unpack(data[4:8])[0]
    header_end = 8 + header_size
    if header_end > len(data):
        raise ValueError("Corrupt LSC header.")
    header = json.loads(data[8:header_end].decode("utf-8"))
    if header.get("version") != VERSION:
        raise ValueError("Unsupported LSC version.")
    raw = zlib.decompress(data[header_end:])
    w, h = int(header["width"]), int(header["height"])
    if len(raw) != w * h * 3:
        raise ValueError("Pixel data size does not match the header.")
    image = Image.frombytes("RGB", (w, h), raw)
    original_size = (int(header.get("original_width", w)), int(header.get("original_height", h)))
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.LANCZOS)
    image.save(output_path)
    return {"input": input_path, "output": output_path, "size": image.size}


def main():
    parser = argparse.ArgumentParser(description="LSC custom lossy image codec")
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
        if args.command == "encode":
            result = encode(args.input, args.output, args.max_kb * 1024)
        else:
            result = decode(args.input, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
