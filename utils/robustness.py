"""
robustness.py — controlled image perturbations for the Robustness Test tab.

Each function takes and returns an RGB uint8 numpy image.
These simulate lighting, orientation, focus and sensor-noise variation.
"""

import cv2
import numpy as np


def adjust_brightness(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """factor 1.0 = unchanged, 0.5 = darker, 1.5 = brighter."""
    if factor == 1.0:
        return img
    return cv2.convertScaleAbs(img, alpha=float(factor), beta=0)


def rotate(img: np.ndarray, angle: float = 0.0) -> np.ndarray:
    """Rotate around the image centre. Empty corners are filled by reflection."""
    if angle == 0:
        return img
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def blur(img: np.ndarray, strength: int = 0) -> np.ndarray:
    """Gaussian blur. strength 0 = off; higher = more out-of-focus."""
    strength = int(strength)
    if strength <= 0:
        return img
    k = strength * 2 + 1  # kernel size must be odd
    return cv2.GaussianBlur(img, (k, k), 0)


def add_noise(img: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Add Gaussian sensor noise. sigma 0 = off; 25 is clearly visible."""
    if sigma <= 0:
        return img
    noise = np.random.normal(0, float(sigma), img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_transforms(img, brightness=1.0, angle=0.0, blur_strength=0, noise_sigma=0.0):
    """Apply all perturbations in a fixed order and return the result."""
    out = np.asarray(img).copy()
    out = adjust_brightness(out, brightness)
    out = rotate(out, angle)
    out = blur(out, blur_strength)
    out = add_noise(out, noise_sigma)
    return out


def describe(brightness=1.0, angle=0.0, blur_strength=0, noise_sigma=0.0) -> str:
    """One-line summary of the applied transforms, for the UI."""
    parts = []
    if brightness != 1.0:
        parts.append(f"brightness ×{brightness:.2f}")
    if angle:
        parts.append(f"rotation {angle:.0f}°")
    if blur_strength:
        parts.append(f"blur k={int(blur_strength) * 2 + 1}")
    if noise_sigma:
        parts.append(f"noise σ={noise_sigma:.0f}")
    return ", ".join(parts) if parts else "no transformation"