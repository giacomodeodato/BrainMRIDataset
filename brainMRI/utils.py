import numpy as np

def preprocess_volume(volume):
    volume = volume.astype(np.float16)
    low = np.percentile(volume, 10, axis=(0, 1, 2))
    high = np.percentile(volume, 99, axis=(0, 1, 2))
    volume = (volume - low) / (high - low)
    return np.clip(volume, 0, 1).astype(np.float16)

def preprocess_mask(mask):
    return np.clip(mask, 0, 1)