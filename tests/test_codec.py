from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import codec


class CodecRegressionTests(unittest.TestCase):
    def test_cached_transform_quantization_matches_wrapper(self):
        rng = np.random.default_rng(7)
        channel = rng.integers(0, 256, size=(19, 23), dtype=np.uint8)
        table = codec._quant_table(codec.QY, 63)

        direct, direct_h, direct_w = codec._dct_quantize(channel, table)
        cached_coeff, cached_h, cached_w = codec._dct_coefficients(channel)
        cached = codec._quantize_coefficients(cached_coeff, table)

        self.assertEqual((direct_h, direct_w), (cached_h, cached_w))
        np.testing.assert_array_equal(direct, cached)

    def test_adaptive_factor_selection_has_three_documented_levels(self):
        rng = np.random.default_rng(12)
        blocks = rng.normal(0, 1, size=(20, 64)).astype(np.float32)
        blocks[:, 0] = 0
        factors = codec._activity_factors(blocks)

        self.assertEqual(factors.shape, (20,))
        self.assertTrue(set(np.unique(factors)).issubset({0, 1, 2}))

        uniform = np.zeros((5, 64), dtype=np.float32)
        np.testing.assert_array_equal(codec._activity_factors(uniform), np.ones(5, dtype=np.uint8))

    def test_encode_decode_round_trip_and_size_limit(self):
        # Smooth gradients are deterministic and compress well enough for a
        # small test cap, without requiring fixture files in the repository.
        yy, xx = np.mgrid[0:96, 0:128]
        rgb = np.stack(
            (
                (xx * 2) % 256,
                (yy * 2) % 256,
                ((xx + yy) * 2) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGB")

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.png"
            encoded = Path(temp) / "image.lsc"
            restored = Path(temp) / "restored.png"
            image.save(source)

            result = codec.encode(str(source), str(encoded), max_bytes=64 * 1024)
            self.assertLessEqual(result["output_bytes"], 64 * 1024)
            self.assertLessEqual(encoded.stat().st_size, 64 * 1024)

            decoded = codec.decode(str(encoded), str(restored))
            self.assertEqual(decoded["size"], image.size)
            self.assertEqual(Image.open(restored).size, image.size)

    def test_rejects_nonpositive_size_limit(self):
        with self.assertRaises(ValueError):
            codec.encode("does-not-need-to-exist.png", "unused.lsc", 0)


if __name__ == "__main__":
    unittest.main()
