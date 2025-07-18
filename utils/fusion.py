"""
fusion.py - Image fusion utilities
"""

import numpy as np
import pywt

def laplacian_fusion_stack(images):
    fused = np.zeros_like(images[0], dtype=np.float32)
    for channel in range(3):
        coeffs = [pywt.dwt2(img[:, :, channel], 'db1') for img in images]
        LLs = [c[0] for c in coeffs]
        LHs = [c[1][0] for c in coeffs]
        HLs = [c[1][1] for c in coeffs]
        HHs = [c[1][2] for c in coeffs]

        fused_LL = np.mean(np.array(LLs), axis=0)
        fused_LH = np.max(np.abs(np.array(LHs)), axis=0)
        fused_HL = np.max(np.abs(np.array(HLs)), axis=0)
        fused_HH = np.max(np.abs(np.array(HHs)), axis=0)

        fused_channel = pywt.idwt2((fused_LL, (fused_LH, fused_HL, fused_HH)), 'db1')
        fused[:, :, channel] = fused_channel[:fused.shape[0], :fused.shape[1]]

    return np.clip(fused, 0, 255).astype(np.uint8)

def gaussian_weighted_stack(images, sigma=1.0):
    import cv2
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    sharpness_maps = [cv2.GaussianBlur(cv2.Laplacian(g, cv2.CV_64F)**2, (3, 3), sigma) for g in gray_images]
    weights = np.stack(sharpness_maps, axis=0)
    weights = weights / (weights.sum(axis=0) + 1e-8)

    weighted_sum = np.zeros_like(images[0], dtype=np.float32)
    for i in range(len(images)):
        for c in range(3):
            weighted_sum[:, :, c] += images[i][:, :, c] * weights[i]

    return np.clip(weighted_sum, 0, 255).astype(np.uint8)
