from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_bitpacked_mask_npz(
    target_file: str | Path,
    mask: np.ndarray,
    *,
    bbox: np.ndarray | None = None,
    source_image: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save a boolean mask to NPZ with bit-packing for maximal storage savings.

    The mask is stored as:
    - mask_packed: uint8 packed bits
    - mask_shape: original shape to restore on load
    """
    mask_bool = np.asarray(mask, dtype=np.bool_)
    flat_mask = mask_bool.reshape(-1)
    packed = np.packbits(flat_mask, bitorder="little")

    payload: dict[str, Any] = {
        "mask_packed": packed,
        "mask_shape": np.asarray(mask_bool.shape, dtype=np.int64),
    }
    if bbox is not None:
        payload["bbox"] = np.asarray(bbox, dtype=np.float32)
    if source_image is not None:
        payload["source_image"] = np.asarray(source_image)
    if extra:
        payload.update(extra)

    np.savez_compressed(target_file, **payload)


def load_bitpacked_mask_npz(target_file: str | Path) -> dict[str, Any]:
    """
    Load a bit-packed mask NPZ file.

    Returns a dict containing:
    - mask: unpacked boolean numpy array
    - any additional stored fields (bbox/source_image/etc)
    """
    with np.load(target_file, allow_pickle=False) as data:
        if "mask_packed" not in data.files or "mask_shape" not in data.files:
            raise ValueError(
                f"{target_file} is missing required bit-packed fields "
                "('mask_packed', 'mask_shape')."
            )

        shape = tuple(int(x) for x in data["mask_shape"].tolist())
        total_size = int(np.prod(shape))
        unpacked = np.unpackbits(data["mask_packed"], bitorder="little")[:total_size]
        mask = unpacked.reshape(shape).astype(np.bool_)

        result: dict[str, Any] = {"mask": mask}
        for key in data.files:
            if key in ("mask_packed", "mask_shape"):
                continue
            value = data[key]
            if isinstance(value, np.ndarray) and value.ndim == 0:
                result[key] = value.item()
            else:
                result[key] = value
        return result
