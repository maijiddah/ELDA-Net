import cv2
import numpy as np

def preprocess_image(image, size):
    image = cv2.resize(image, tuple(size))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return image.astype(np.float32)
