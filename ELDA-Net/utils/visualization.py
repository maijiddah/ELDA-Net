import cv2
import numpy as np

def overlay_lanes(frame, mask):
    overlay = frame.copy()
    overlay[mask > 0.5] = [0, 255, 0]
    return overlay
