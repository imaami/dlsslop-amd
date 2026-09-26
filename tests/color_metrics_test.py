#!/usr/bin/env python3
"""Run with python3 tests/color_metrics_test.py; needs numpy and Pillow."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "color_metrics.py"
SPEC = importlib.util.spec_from_file_location("color_metrics", MODULE)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class ColorMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def raw(self, name, data):
        path = self.directory / name
        path.write_bytes(data)
        return path

    def test_raw_rgba_and_bgra_and_packed8(self):
        expected = np.array([[[10, 20, 30]]]) / 255
        for vk_format, raw in ((37, bytes([10, 20, 30, 255])),
                               (44, bytes([30, 20, 10, 255])),
                               (51, bytes([10, 20, 30, 255]))):
            rgb = metrics.load_capture(self.raw("capture.raw", raw), vk_format, 1, 1)
            np.testing.assert_array_equal(rgb, expected)

    def test_png_bgra_is_not_swizzled_twice(self):
        path = self.directory / "capture.png"
        Image.fromarray(np.array([[[11, 99, 217, 255]]], dtype=np.uint8)).save(path)
        actual = metrics.load_capture(path, 44, 1, 1, "png")
        np.testing.assert_array_equal(actual, np.array([[[11, 99, 217]]]) / 255)

    def test_packed10_component_layout(self):
        red, green, blue = 1023, 511, 7
        for vk_format, packed in ((64, red | green << 10 | blue << 20 | 3 << 30),
                                  (58, blue | green << 10 | red << 20 | 3 << 30)):
            path = self.raw("capture.raw", int(packed).to_bytes(4, "little"))
            np.testing.assert_array_equal(metrics.load_capture(path, vk_format, 1, 1),
                                          np.array([[[red, green, blue]]]) / 1023)

    def test_fp16_preserves_hdr_and_negative_values(self):
        rgba = np.array([[[-0.25, 2, 7.5, 1]]], dtype="<f2")
        actual = metrics.load_capture(self.raw("capture.raw", rgba.tobytes()), 97, 1, 1)
        np.testing.assert_array_equal(actual, rgba[..., :3])

    def test_bad_dimensions_and_payloads_are_rejected(self):
        path = self.raw("capture.raw", b"\0" * 3)
        with self.assertRaises(ValueError):
            metrics.load_capture(path, 37, 1, 1)
        with self.assertRaises(ValueError):
            metrics.load_capture(path, 37, -1, 1)
        with self.assertRaisesRegex(ValueError, "unsupported Vulkan format: 99"):
            metrics.load_capture(path, 99, 1, 1)

    def test_pfm_endianness_orientation_and_scale(self):
        expected = np.arange(12).reshape(2, 2, 3).astype(float)
        for order, scale in (("<", -2), (">", 2)):
            payload = np.flipud(expected / 2).astype(order + "f4").tobytes()
            path = self.raw("test.pfm", f"PF\n2 2\n{scale}\n".encode() + payload)
            np.testing.assert_array_equal(metrics.read_pfm(path), expected)

    def test_identity_and_channel_swap_candidates(self):
        rgb = np.array([[[0.1, 0.5, 0.9], [0.3, 0.7, 0.2]]])
        self.assertTrue(metrics.diagnose_identity(rgb, rgb)["passed"])
        result = metrics.diagnose_identity(rgb, rgb[..., ::-1])
        self.assertFalse(result["passed"])
        self.assertIn("channel_order_BGR", [x["transformation"] for x in result["candidates"]])

    def test_srgb_decode_and_encode_candidates(self):
        rgb = np.array([[[0.01, 0.3, 0.8], [0.001, 0.5, 0.7]]])
        decoded = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        encoded = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
        for actual, name in ((decoded, "unexpected_srgb_decode"), (encoded, "unexpected_srgb_encode")):
            result = metrics.diagnose_identity(rgb, actual)
            self.assertIn(name, [x["transformation"] for x in result["candidates"]])

    def test_neutral_red_bias_and_achromatic_gain_are_distinguished(self):
        rgb = np.full((2, 3, 3), 0.3)
        warm = rgb.copy()
        warm[..., 0] += 0.1
        result = metrics.compare_stats(rgb, warm)
        self.assertAlmostEqual(result["red_excess_delta"], 0.1)
        self.assertEqual(result["reference_neutral_chromaticity"]["pixels"], 6)
        self.assertGreater(result["reference_neutral_chromaticity"]["mean_channel_delta"]["r"], 0)
        gain = metrics.compare_stats(rgb, rgb * 2)
        self.assertAlmostEqual(gain["chromaticity"]["mean_absolute_delta"], 0)
        self.assertNotIn("passed", result)

    def test_empty_eligibility_and_nonfinite_reports_are_json_safe(self):
        for rgb in (np.zeros((1, 1, 3)), np.full((1, 1, 3), np.nan),
                    np.array([[[np.inf, -np.inf, 1]]])):
            with np.errstate(invalid="ignore"):
                json.dumps(metrics.image_stats(rgb), allow_nan=False)
                json.dumps(metrics.compare_stats(rgb, rgb), allow_nan=False)
                json.dumps(metrics.diagnose_identity(rgb, rgb), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
