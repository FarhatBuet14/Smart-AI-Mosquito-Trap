"""
deblur.py - Classical and deep learning-based deblurring methods
"""

import cv2
import numpy as np
from skimage.restoration import wiener

def wiener_deblur_rgb(img, kernel_size=7):
    psf = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    channels = cv2.split(img)
    deblurred_channels = []

    for c in channels:
        c_norm = c.astype(np.float32) / 255.0
        c_deblur = wiener(c_norm, psf, balance=0.01)
        c_deblur = np.clip(c_deblur, 0, 1)
        c_deblur = (c_deblur * 255).astype(np.uint8)
        deblurred_channels.append(c_deblur)

    return cv2.merge(deblurred_channels)

def sharpen_unsharp_mask(img, amount=1.0, radius=3):
    blurred = cv2.GaussianBlur(img, (radius, radius), 0)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
