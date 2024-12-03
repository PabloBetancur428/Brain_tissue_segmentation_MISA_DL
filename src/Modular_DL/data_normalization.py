import numpy as np

def z_score_normalize(image):

    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image