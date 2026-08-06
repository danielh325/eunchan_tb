"""Bone suppression preprocessing step.

Evidence: VGG-16 on bone-suppressed CXRs beat the same model on non-suppressed
images by 5-11 AUC points (statistically significant, p<0.05) on Shenzhen/
Montgomery TB benchmarks -- a real, cheap, backbone-orthogonal lever that
wasn't in the baseline notebook or the user's own SSRN paper. Ribs/clavicles
are also acquisition-geometry-correlated (their apparent contrast/position
shifts with scanner and patient positioning), so suppressing them plausibly
removes some of the same shortcut signal that lung-ROI cropping targets.

No pretrained bone-suppression network is vendored here (would need a paired
bone/soft-tissue training set, e.g. JSRT's dual-energy-derived ground truth,
which isn't available in this repo) -- ships a classical, dependency-free
homomorphic-filtering approximation instead: bones are the dominant
low-frequency, high-intensity structures in a CXR, so a large-kernel
Gaussian-estimated "bone layer" subtracted from the image suppresses rib/
clavicle contrast while leaving lung texture (higher-frequency) largely
intact. This is a real, if weaker, approximation of the cited method's
effect -- flagged as an approximation, not a re-implementation of a trained
bone-suppression model, and worth ablating (`--use-bone-suppression` on/off)
rather than assumed to always help.
"""
import cv2
import numpy as np


def suppress_bones(u8: np.ndarray, kernel_frac: float = 0.08, strength: float = 0.5) -> np.ndarray:
    """u8: (H, W) uint8 grayscale, already CLAHE'd. Returns uint8 (H, W)."""
    h, w = u8.shape
    k = max(3, int(min(h, w) * kernel_frac) | 1)  # odd kernel size
    bone_layer = cv2.GaussianBlur(u8, (k, k), 0).astype(np.float32)
    img = u8.astype(np.float32)
    suppressed = img - strength * (bone_layer - img.mean())
    return np.clip(suppressed, 0, 255).astype(np.uint8)
