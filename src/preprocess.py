"""Preprocessing shared by dataset building, training and the Streamlit app.

Everything works on RGB uint8 numpy arrays (H, W, 3). Use the same functions
at training time and at inference time, otherwise accuracy drops in deployment.
"""
import cv2
import numpy as np
from PIL import Image

# ---- tunable defaults (change in ONE place, then rebuild the dataset) ----
TARGET_MEAN = 0.42     # brightness the adaptive gamma aims for (0-1)
CUTOFF = 1.0           # percentile clipped at each end by the contrast stretch
CLAHE_CLIP = 2.0
CLAHE_TILES = 8
EDGE_BLUR = 1.2        # Gaussian sigma before Sobel-Feldman (suppresses bag wrinkles)

VARIANTS = ("raw", "enh", "cs3")


def to_rgb_array(img):
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    return np.asarray(img)


def enhance_rgb(rgb, target_mean=TARGET_MEAN, cutoff=CUTOFF):
    """Percentile contrast stretch (shared across channels) + adaptive gamma.

    The gamma is chosen per crop so dark photos and bright photos end up with a
    similar overall brightness (instead of one fixed gamma for everything)."""
    rgb = to_rgb_array(rgb)
    # one shared stretch for all channels: keeps hue/saturation, so the belt colour
    # (and the colour mask) stays comparable between dark and bright photos
    lo = np.float32(np.percentile(rgb, cutoff))
    hi = np.float32(np.percentile(rgb, 100 - cutoff))
    x = np.clip((rgb.astype(np.float32) - lo) / max(float(hi - lo), 1.0), 0, 1)
    mean = float(x.mean())
    if 1e-3 < mean < 0.999:
        gamma = float(np.clip(np.log(target_mean) / np.log(mean), 0.4, 1.0))
        x = x ** gamma
    return (x * 255).astype(np.uint8)


def clahe_lightness(rgb_enh, clip=CLAHE_CLIP, tiles=CLAHE_TILES):
    """CLAHE on the L channel: evens out dark bands and glare."""
    lab = cv2.cvtColor(rgb_enh, cv2.COLOR_RGB2LAB)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])


def edge_magnitude(gray, blur=EDGE_BLUR):
    """Sobel-Feldman gradient magnitude (OpenCV Scharr = optimised Feldman kernel),
    normalised per crop. High value = strong edge."""
    g = cv2.GaussianBlur(gray, (0, 0), blur) if blur > 0 else gray
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag / max(float(np.percentile(mag, 99.5)), 1e-6), 0, 1)
    return (mag * 255).astype(np.uint8)


def belt_mask_rgb(rgb_enh):
    """Yellow/orange belt mask on the brightness-normalised crop (0 or 255)."""
    hsv = np.asarray(Image.fromarray(rgb_enh).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= 18) & (h <= 50) & (s >= 70) & (v >= 70)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def make_variants(img):
    """Return every model input variant for one crop, plus the parts for previews.

    raw : the crop as saved by 01_make_crops.py
    enh : contrast-stretched, brightness-normalised RGB
    cs3 : [CLAHE lightness, Sobel-Feldman edges, belt mask] as 3 channels
    """
    raw = to_rgb_array(img)
    enh = enhance_rgb(raw)
    light = clahe_lightness(enh)
    edge = edge_magnitude(light)
    mask = belt_mask_rgb(enh)
    return {
        "raw": raw, "enh": enh,
        "cs3": np.dstack([light, edge, mask]),
        "parts": {"clahe": light, "edge": edge, "mask": mask},
    }
