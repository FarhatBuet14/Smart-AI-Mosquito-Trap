"""
align.py - ECC-based image alignment
"""

import cv2
import numpy as np

def align_images(reference_img, image_to_align):
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
    try:
        cc, warp_matrix = cv2.findTransformECC(
            cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY),
            warp_matrix, warp_mode, criteria
        )
        aligned_img = cv2.warpAffine(image_to_align, warp_matrix,
                                     (image_to_align.shape[1], image_to_align.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_img
    except cv2.error:
        return image_to_align
