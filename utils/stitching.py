from __future__ import annotations

from typing import Callable

import torch


def compute_tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return tile start indices that cover a 1D axis end-to-end.

    The final start index is forced to ``length - tile_size`` when needed so
    the far edge is always covered, even when that value is not an exact
    multiple of ``stride``.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _build_center_weight(tile_size: int, min_weight: float) -> torch.Tensor:
    """Create a smooth 2D center-priority blending window for one tile.

    Borders are down-weighted and centers are up-weighted to reduce seams in
    overlap regions. A non-zero floor is applied to avoid zero-denominator
    regions during normalization.
    """
    coords = torch.arange(tile_size, dtype=torch.float32)
    dist = torch.minimum(coords, (tile_size - 1) - coords)
    one_d = (dist + 1.0) / (tile_size / 2.0)
    one_d = torch.clamp(one_d, min=min_weight)
    return one_d[:, None] * one_d[None, :]


def stitch_overlapping_tile_probs(
    image_tensor: torch.Tensor,
    predict_tiles: Callable[[torch.Tensor], torch.Tensor],
    num_classes: int,
    tile_size: int = 256,
    stride: int = 128,
    batch_size: int = 1,
    min_weight: float = 0.05,
    include_tile: Callable[[int, int, int, int], bool] | None = None,
    on_batch_processed: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """Run tiled inference and stitch overlap regions with normalized blending.

    Parameters
    ----------
    image_tensor:
        Input image tensor of shape ``[C, H, W]``.
    predict_tiles:
        Callable that receives tiles ``[B, C, tile_size, tile_size]`` and
        returns probabilities ``[B, num_classes, tile_size, tile_size]``.
    num_classes:
        Number of output channels to stitch.
    tile_size:
        Extraction/prediction window size.
    stride:
        Step between tile starts. Values below ``tile_size`` create overlap.
    batch_size:
        Number of tiles per prediction call.
    min_weight:
        Lower bound for blending weights. Must be ``> 0`` and ``<= 1``.
    include_tile:
        Optional filter predicate called as ``(y0, x0, valid_h, valid_w)``.
        Return ``False`` to skip a tile (e.g., known whitespace).
    on_batch_processed:
        Optional progress callback called with ``1`` for each processed batch.

    Returns
    -------
    torch.Tensor
        Stitched probabilities of shape ``[num_classes, H, W]``.

    How overlap stitching works
    ---------------------------
    The merge is a weighted overlap-add followed by per-pixel normalization:

    - For each tile ``i``, compute a spatial window ``w_i`` (higher in center,
      lower at borders).
    - Accumulate weighted predictions:
      ``accum += p_i * w_i``.
    - Accumulate weights:
      ``weight_accum += w_i``.
    - Normalize per pixel:
      ``merged = accum / weight_accum``.

    In overlap zones, this is a proper weighted average because each tile's
    *effective* contribution is ``w_i / sum_j(w_j)`` at that pixel.

    Why this is robust
    ------------------
    - Reduces seam artifacts by de-emphasizing less reliable tile borders.
    - Avoids intensity/probability drops in overlaps by explicit normalization.
    - Remains stable when some tiles are skipped (local renormalization keeps
      values on a consistent scale).

    Artifact expectations
    ---------------------
    This approach usually removes hard seams. The tradeoff is mild smoothing in
    regions where neighboring tiles disagree strongly. In practice this is often
    preferable to abrupt stitch boundaries.
    """
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image_tensor [C,H,W], got shape={tuple(image_tensor.shape)}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if not (0.0 < min_weight <= 1.0):
        raise ValueError(f"min_weight must be in (0,1], got {min_weight}")

    _, image_h, image_w = image_tensor.shape
    y_starts = compute_tile_starts(image_h, tile_size, stride)
    x_starts = compute_tile_starts(image_w, tile_size, stride)

    tile_weight = _build_center_weight(tile_size, min_weight=min_weight)
    accum = torch.zeros((num_classes, image_h, image_w), dtype=torch.float32)
    weight_accum = torch.zeros((image_h, image_w), dtype=torch.float32)

    pending_tiles: list[torch.Tensor] = []
    pending_coords: list[tuple[int, int, int, int]] = []

    def process_batch(
        batch_tiles: list[torch.Tensor], batch_coords: list[tuple[int, int, int, int]]
    ) -> None:
        if not batch_tiles:
            return

        batch = torch.stack(batch_tiles, dim=0)
        pred_probs = predict_tiles(batch).to(torch.float32)
        if pred_probs.ndim != 4:
            raise RuntimeError(f"Expected predictions [B,K,T,T], got ndim={pred_probs.ndim}")
        if pred_probs.shape[1] != num_classes:
            raise RuntimeError(
                f"Expected {num_classes} classes, got prediction shape {tuple(pred_probs.shape)}"
            )

        for idx, (y0, x0, valid_h, valid_w) in enumerate(batch_coords):
            y1 = y0 + valid_h
            x1 = x0 + valid_w
            local_weight = tile_weight[:valid_h, :valid_w]
            accum[:, y0:y1, x0:x1] += pred_probs[idx, :, :valid_h, :valid_w] * local_weight
            weight_accum[y0:y1, x0:x1] += local_weight

        if on_batch_processed is not None:
            on_batch_processed(1)

    def flush_full_batches() -> None:
        while len(pending_tiles) >= batch_size:
            batch_tiles = pending_tiles[:batch_size]
            batch_coords = pending_coords[:batch_size]
            del pending_tiles[:batch_size]
            del pending_coords[:batch_size]
            process_batch(batch_tiles, batch_coords)

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, image_h)
            x1 = min(x0 + tile_size, image_w)
            valid_h = y1 - y0
            valid_w = x1 - x0

            if include_tile is not None and not include_tile(y0, x0, valid_h, valid_w):
                continue

            tile = torch.zeros((image_tensor.shape[0], tile_size, tile_size), dtype=torch.float32)
            tile[:, :valid_h, :valid_w] = image_tensor[:, y0:y1, x0:x1]

            pending_tiles.append(tile)
            pending_coords.append((y0, x0, valid_h, valid_w))
            flush_full_batches()

    process_batch(pending_tiles, pending_coords)

    safe_den = torch.clamp(weight_accum, min=1e-6).unsqueeze(0)
    return accum / safe_den
