import numpy as np

ALBU_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ALBU_DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_rgb_albu_defaults(image_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Normalize an RGB image using albumentations Normalize() default statistics.

    Expects image shape [H, W, 3] with uint8-like values in [0, 255].
    Returns float32 array with shape [H, W, 3]:
        (x / 255 - mean) / std
    """
    image_np = image_rgb_uint8.astype(np.float32) / 255.0
    return (image_np - ALBU_DEFAULT_MEAN) / ALBU_DEFAULT_STD
