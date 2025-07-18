"""
undistort.py - Fisheye distortion correction
"""

import cv2
import numpy as np

def remove_fisheye_distortion(img):
    h, w = img.shape[:2]
    K = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]])
    D = np.array([-0.3, 0.1, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
