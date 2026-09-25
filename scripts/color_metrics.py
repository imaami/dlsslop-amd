"""Stage-local RGB measurements; color changes alone do not prove model failure.

Arrays retain their capture domain. UNORM formats use normalized code values;
FP16 is not clipped or tone mapped. Cross-domain comparisons are invalid.
"""

from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image


_FORMATS = {
    37: ("R8G8B8A8_UNORM", 4, "normalized_codes_transfer_unspecified"),
    44: ("B8G8R8A8_UNORM", 4, "normalized_codes_transfer_unspecified"),
    51: ("A8B8G8R8_UNORM_PACK32", 4, "normalized_codes_transfer_unspecified"),
    58: ("A2R10G10B10_UNORM_PACK32", 4, "normalized_codes_transfer_unspecified"),
    64: ("A2B10G10R10_UNORM_PACK32", 4, "normalized_codes_transfer_unspecified"),
    97: ("R16G16B16A16_SFLOAT", 8, "native_float_transfer_unspecified"),
}


def format_info(vk_format):
    """A VkFormat identifies storage, not its swapchain color space."""
    try:
        name, size, domain = _FORMATS[int(vk_format)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unsupported Vulkan format: {vk_format}") from error
    return {"name": name, "bytes_per_pixel": size, "numeric_domain": domain,
            "transfer_function": "unknown_without_color_space_metadata"}


def load_capture(path, vk_format, width, height, encoding=None):
    """Read a tightly packed capture. PNG channels are already canonical RGB.

    Raw byte order matches the little-endian AMD Linux capture producer. The
    encoding argument describes the file container (png/raw), not gamma/PQ.
    """
    path = Path(path)
    info = format_info(vk_format)
    width, height, vk_format = int(width), int(height), int(vk_format)
    if width <= 0 or height <= 0:
        raise ValueError("capture dimensions must be positive")
    encoding = (encoding or path.suffix.lstrip(".")).lower()
    if encoding == "png":
        if vk_format not in (37, 44, 51):
            raise ValueError("PNG capture is only defined for 8-bit RGBA formats")
        with Image.open(path) as im:
            if im.size != (width, height):
                raise ValueError("PNG dimensions differ from capture manifest")
            return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    if encoding != "raw":
        raise ValueError(f"unsupported capture encoding: {encoding}")
    data = path.read_bytes()
    expected = width * height * info["bytes_per_pixel"]
    if len(data) != expected:
        raise ValueError(f"raw capture size {len(data)} differs from expected {expected}")
    if vk_format == 97:
        return np.frombuffer(data, dtype="<f2").reshape(height, width, 4)[..., :3].astype(np.float64)
    if vk_format in (58, 64):
        packed = np.frombuffer(data, dtype="<u4").reshape(height, width)
        lo, mid, hi = packed & 1023, (packed >> 10) & 1023, (packed >> 20) & 1023
        channels = (hi, mid, lo) if vk_format == 58 else (lo, mid, hi)
        return np.stack(channels, axis=-1).astype(np.float64) / 1023.0
    rgba = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    indices = [2, 1, 0] if vk_format == 44 else [0, 1, 2]
    return rgba[..., indices].astype(np.float64) / 255.0


def read_pfm(path):
    """Read RGB PFMs, including bottom-up ordering and nonunit scale."""
    with open(path, "rb") as stream:
        def line():
            while True:
                value = stream.readline()
                if not value:
                    raise ValueError("truncated PFM header")
                value = value.strip()
                if value and not value.startswith(b"#"):
                    return value
        if line() != b"PF":
            raise ValueError("expected RGB PFM (PF)")
        dimensions = line().split()
        if len(dimensions) != 2:
            raise ValueError("invalid PFM dimensions")
        width, height = map(int, dimensions)
        scale = float(line())
        if width <= 0 or height <= 0 or not np.isfinite(scale) or not scale:
            raise ValueError("invalid PFM dimensions or scale")
        data = stream.read()
        if len(data) != width * height * 12:
            raise ValueError("PFM payload size differs from header")
        rgb = np.frombuffer(data, dtype="<f4" if scale < 0 else ">f4")
        return np.flipud(rgb.reshape(height, width, 3)).astype(np.float64) * abs(scale)


def _rgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("expected a nonempty HxWx3 RGB array")
    return rgb


def _channels(values):
    return {name: float(value) if np.isfinite(value) else None
            for name, value in zip(("r", "g", "b"), values)}


def _eligible(rgb):
    # This is a code-domain measurement window, not an HDR luminance test.
    return (np.isfinite(rgb).all(axis=-1) & (rgb.min(axis=-1) >= 0.02)
            & (rgb.max(axis=-1) <= 0.98))


def _neutral(rgb, eligible):
    return eligible & ((rgb.max(axis=-1) - rgb.min(axis=-1))
                       <= 0.05 * np.maximum(rgb.sum(axis=-1), 1e-12))


def image_stats(rgb):
    rgb = _rgb(rgb)
    finite = np.isfinite(rgb).all(axis=-1)
    pixels = rgb[finite]
    total = int(finite.size)
    eligible = _eligible(rgb)
    result = {
        "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
        "pixels": total, "nonfinite_pixels": int((~finite).sum()),
        "unit_interval_eligible_pixels": int(eligible.sum()),
        "neutral_eligible_pixels": int(_neutral(rgb, eligible).sum()),
        "eligibility": "all reference channels 0.02..0.98; neutral spread <= 0.05*RGB sum",
        "endpoint_note": "Endpoint occupancy is not proof of clipping; HDR may legitimately exceed 1 or have negative channels.",
    }
    for name, fn in (("mean", np.mean), ("min", np.min), ("max", np.max)):
        result[name] = _channels(fn(pixels, axis=0) if pixels.size else [np.nan] * 3)
    result["mean_red_excess"] = float(np.mean(pixels[:, 0] - pixels[:, 1:].mean(axis=1))) if pixels.size else None
    result["below_zero_samples"] = _channels((pixels < 0).sum(axis=0))
    result["above_one_samples"] = _channels((pixels > 1).sum(axis=0))
    result["at_or_below_zero_percent"] = _channels((pixels <= 0).sum(axis=0) * 100.0 / total)
    result["at_or_above_one_percent"] = _channels((pixels >= 1).sum(axis=0) * 100.0 / total)
    return result


def compare_stats(reference, result):
    reference, result = _rgb(reference), _rgb(result)
    if reference.shape != result.shape:
        raise ValueError("comparison dimensions differ")
    finite = np.isfinite(reference).all(axis=-1) & np.isfinite(result).all(axis=-1)
    delta = result[finite] - reference[finite]
    ref_eligible = _eligible(reference)
    usable = (ref_eligible & finite & (result.min(axis=-1) >= 0)
              & (result.sum(axis=-1) > 0.06))
    neutral = usable & _neutral(reference, ref_eligible)
    report = {
        "interpretation": "Measurements only. A neural model may intentionally change color or illumination.",
        "finite_paired_pixels": int(finite.sum()), "invalid_paired_pixels": int((~finite).sum()),
        "mean_channel_delta": _channels(delta.mean(axis=0) if delta.size else [np.nan] * 3),
        "mean_absolute_channel_delta": _channels(np.abs(delta).mean(axis=0) if delta.size else [np.nan] * 3),
        "max_absolute_channel_delta": _channels(np.abs(delta).max(axis=0) if delta.size else [np.nan] * 3),
        "red_excess_delta": float(np.mean(delta[:, 0] - delta[:, 1:].mean(axis=1))) if delta.size else None,
        "changed_pixel_threshold": 1e-6,
        "changed_pixels_percent": float(np.mean(np.abs(delta).max(axis=1) > 1e-6) * 100) if delta.size else None,
        "new_upper_endpoint_pixels": int((finite & (reference.max(axis=-1) < 1) & (result.max(axis=-1) >= 1)).sum()),
        "new_lower_endpoint_pixels": int((finite & (reference.min(axis=-1) > 0) & (result.min(axis=-1) <= 0)).sum()),
        "endpoint_note": "New endpoint counts are not proof of clipping and are not HDR clipping tests.",
    }
    for name, mask in (("chromaticity", usable), ("reference_neutral_chromaticity", neutral)):
        ref, out = reference[mask], result[mask]
        change = out / out.sum(axis=-1, keepdims=True) - ref / ref.sum(axis=-1, keepdims=True) if ref.size else np.empty((0, 3))
        report[name] = {
            "pixels": int(mask.sum()),
            "mean_channel_delta": _channels(change.mean(axis=0) if change.size else [np.nan] * 3),
            "mean_absolute_delta": float(np.abs(change).mean()) if change.size else None,
        }
    return report


def diagnose_identity(reference, result, tolerance=1 / 255 + 1e-7):
    """Analytical checks ONLY for a path which is expected to be identity.

    Candidate transformations are clues, never declarations of a root cause.
    Do not apply the pass/fail interpretation to normal neural rendering.
    """
    reference, result = _rgb(reference), _rgb(result)
    if reference.shape != result.shape:
        raise ValueError("identity comparison dimensions differ")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("identity tolerance must be finite and nonnegative")
    finite = np.isfinite(reference).all(axis=-1) & np.isfinite(result).all(axis=-1)
    ref, out = reference[finite], result[finite]
    report = {"expected_identity": True, "tolerance": float(tolerance),
              "invalid_paired_pixels": int((~finite).sum()), "candidates": []}
    if not ref.size:
        report.update(passed=False, rmse=None, max_absolute_error=None)
        return report
    error = out - ref
    baseline = float(np.sqrt(np.mean(error * error)))
    maximum = float(np.abs(error).max())
    report.update(passed=bool(finite.all() and maximum <= tolerance),
                  rmse=baseline, max_absolute_error=maximum)
    if report["passed"]:
        return report
    def candidate(name, expected):
        residual = float(np.sqrt(np.mean((out - expected) ** 2)))
        if residual <= tolerance and residual < baseline * 0.2:
            report["candidates"].append({"transformation": name, "rmse": residual,
                                          "note": "Matches an analytical hypothesis; confirm the pipeline stage."})
    for order in permutations(range(3)):
        if order != (0, 1, 2):
            candidate("channel_order_" + "".join("RGB"[i] for i in order), ref[:, order])
    if np.all((ref >= 0) & (ref <= 1)):
        srgb_decode = np.where(ref <= 0.04045, ref / 12.92, ((ref + 0.055) / 1.055) ** 2.4)
        srgb_encode = np.where(ref <= 0.0031308, ref * 12.92, 1.055 * ref ** (1 / 2.4) - 0.055)
        candidate("unexpected_srgb_decode", srgb_decode)
        candidate("unexpected_srgb_encode", srgb_encode)
    return report
