from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from commands.view_image import _resolve_channel_indices
from utils.model_utils import MODEL_OUTPUT_CHANNEL_LABELS
from utils.npz_utils import load_bitpacked_mask_npz


Image.MAX_IMAGE_PIXELS = None


def _fit_size(width: int, height: int, max_dim: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 1, 1
    scale = min(max_dim / float(max(width, height)), 1.0)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return out_w, out_h


def _safe_bbox_to_preview_pixels(
    bbox: dict[str, Any], preview_w: int, preview_h: int
) -> tuple[int, int, int, int]:
    x0 = int(round(float(bbox["x_norm"]) * preview_w))
    y0 = int(round(float(bbox["y_norm"]) * preview_h))
    w = int(round(float(bbox["w_norm"]) * preview_w))
    h = int(round(float(bbox["h_norm"]) * preview_h))

    x0 = min(max(0, x0), preview_w)
    y0 = min(max(0, y0), preview_h)
    x1 = min(preview_w, x0 + max(1, w))
    y1 = min(preview_h, y0 + max(1, h))
    return x0, y0, x1, y1


def _extract_index_from_name(name: str, slide_prefix: str) -> int | None:
    # Matches names like "1_3.png" for slide prefix "1".
    m = re.fullmatch(rf"{re.escape(slide_prefix)}_(\d+)\.png", name)
    if m is None:
        return None
    return int(m.group(1))


class _ViewerApp:
    def __init__(
        self,
        pil_images: list[Image.Image],
        labels: list[str],
        max_dim: int,
        extract_boxes: list[tuple[str, tuple[int, int, int, int]]] | None = None,
    ) -> None:
        self.pil_images = [im.convert("RGB") for im in pil_images]
        self.labels = labels
        self.max_dim = max_dim
        self.extract_boxes = extract_boxes or []
        self.fig: plt.Figure | None = None
        self._main_ax: plt.Axes | None = None
        self._footer_text: Any | None = None
        self._extract_boxes_resized: list[tuple[str, tuple[float, float, float, float]]] = []

    def _compute_grid(self, n_items: int) -> tuple[int, int]:
        cols = max(1, math.ceil(math.sqrt(n_items)))
        rows = math.ceil(n_items / cols)
        return rows, cols

    def _get_resized_rgb_arrays(self) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        for img in self.pil_images:
            out_w, out_h = _fit_size(img.width, img.height, self.max_dim)
            resized = img.resize((out_w, out_h), resample=Image.Resampling.NEAREST)
            arrays.append(np.asarray(resized))
        return arrays

    def _update_visible_extracts_label(self) -> None:
        if self._main_ax is None or self._footer_text is None or self.fig is None:
            return

        xlim = self._main_ax.get_xlim()
        ylim = self._main_ax.get_ylim()
        view_x0, view_x1 = min(xlim), max(xlim)
        view_y0, view_y1 = min(ylim), max(ylim)

        visible: list[str] = []
        for extract_id, (x0, y0, x1, y1) in self._extract_boxes_resized:
            intersects = not (x1 < view_x0 or x0 > view_x1 or y1 < view_y0 or y0 > view_y1)
            if intersects:
                visible.append(extract_id)

        if not visible:
            footer = "Visible stitched extracts: none"
        else:
            display_limit = 18
            suffix = ""
            if len(visible) > display_limit:
                suffix = f", ... (+{len(visible) - display_limit} more)"
            footer = (
                f"Visible stitched extracts ({len(visible)}): "
                + ", ".join(visible[:display_limit])
                + suffix
            )

        self._footer_text.set_text(footer)
        self.fig.canvas.draw_idle()

    def _on_axes_limits_changed(self, _ax: Any) -> None:
        self._update_visible_extracts_label()

    def _render(self) -> None:
        arrays = self._get_resized_rgb_arrays()
        n_items = len(arrays)
        rows, cols = self._compute_grid(n_items)
        fig_w = max(8.0, cols * 3.2)
        fig_h = max(6.0, rows * 3.0 + 0.8)
        self.fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(fig_w, fig_h),
            sharex=True,
            sharey=True,
        )
        self.fig.canvas.manager.set_window_title("GigaTIME Whole-Slide Viewer")
        self.fig.patch.set_facecolor("#1f1f1f")
        bottom_margin = 0.12 if not self.extract_boxes else 0.16
        self.fig.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.95,
            bottom=bottom_margin,
            wspace=0.08,
            hspace=0.22,
        )

        axes_flat = np.atleast_1d(axes).ravel()
        for ax in axes_flat:
            ax.set_facecolor("#1f1f1f")

        for idx, (array, label) in enumerate(zip(arrays, self.labels)):
            ax = axes_flat[idx]
            ax.imshow(array, interpolation="nearest")
            ax.set_title(label, color="#f5f5f5", fontsize=11, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for idx in range(n_items, len(axes_flat)):
            axes_flat[idx].axis("off")

        self._main_ax = axes_flat[0]
        if self.extract_boxes:
            out_w, out_h = arrays[0].shape[1], arrays[0].shape[0]
            src_w, src_h = self.pil_images[0].width, self.pil_images[0].height
            scale_x = out_w / float(max(1, src_w))
            scale_y = out_h / float(max(1, src_h))
            self._extract_boxes_resized = [
                (
                    extract_id,
                    (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                )
                for extract_id, (x0, y0, x1, y1) in self.extract_boxes
            ]

            self._footer_text = self.fig.text(
                0.5,
                0.03,
                "",
                ha="center",
                va="center",
                color="#f5f5f5",
                fontsize=9,
            )
            self._main_ax.callbacks.connect("xlim_changed", self._on_axes_limits_changed)
            self._main_ax.callbacks.connect("ylim_changed", self._on_axes_limits_changed)
            self._update_visible_extracts_label()

    def run(self) -> None:
        self._render()
        if self.fig is None:
            return
        plt.show()


@click.command("view-whole-slide")
@click.argument("preview_png_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("channels", required=False, default="DAPI")
@click.option(
    "--max-dim",
    type=int,
    default=5000,
    show_default=True,
    help="Maximum size (pixels) of the larger side for each image before layout scaling.",
)
def view_whole_slide(preview_png_file: Path, channels: str, max_dim: int) -> None:
    """
    Open a viewer showing a WSI preview and stitched channel masks for that slide.

    The command expects a preview filename like "<slide>_preview.png" and stitches
    masks from matching "<slide>_<index>.npz" files using bounding boxes.
    """
    if preview_png_file.suffix.lower() != ".png":
        raise click.ClickException(f"Expected a .png file, got: {preview_png_file}")
    if not preview_png_file.name.endswith("_preview.png"):
        raise click.ClickException(
            "Expected preview file name to end with '_preview.png', "
            f"got: {preview_png_file.name}"
        )
    if max_dim <= 0:
        raise click.ClickException("--max-dim must be > 0")

    patient_folder = preview_png_file.parent
    bbox_path = patient_folder / "bounding_boxes.json"
    if not bbox_path.exists():
        raise click.ClickException(f"Missing required file: {bbox_path}")

    slide_prefix = preview_png_file.stem[: -len("_preview")]
    with bbox_path.open("r", encoding="utf-8") as f:
        bbox_index: dict[str, dict[str, Any]] = json.load(f)

    slide_entries: list[tuple[int, str, dict[str, Any]]] = []
    for image_name, bbox in bbox_index.items():
        index = _extract_index_from_name(image_name, slide_prefix)
        if index is None:
            continue
        slide_entries.append((index, image_name, bbox))

    if not slide_entries:
        raise click.ClickException(
            f"No bounding box entries found for slide '{slide_prefix}' in {bbox_path}"
        )
    slide_entries.sort(key=lambda item: item[0])

    first_npz_path = patient_folder / f"{slide_prefix}_{slide_entries[0][0]}.npz"
    if not first_npz_path.exists():
        raise click.ClickException(
            "Could not determine channel count because no NPZ file exists for "
            f"slide '{slide_prefix}'. Expected at least: {first_npz_path}"
        )

    first_npz = load_bitpacked_mask_npz(first_npz_path)
    if "mask" not in first_npz:
        raise click.ClickException(f"Missing 'mask' in NPZ file: {first_npz_path}")

    first_mask = np.asarray(first_npz["mask"])
    if first_mask.ndim == 2:
        first_mask = first_mask[np.newaxis, :, :]
    if first_mask.ndim != 3:
        raise click.ClickException(
            f"Expected mask shape [C,H,W] or [H,W] in {first_npz_path}, got {first_mask.shape}"
        )

    mask_channels = int(first_mask.shape[0])
    channel_specs = _resolve_channel_indices(mask_channels, channels)

    with Image.open(preview_png_file) as img:
        preview_image = img.convert("RGB")
    preview_w, preview_h = preview_image.width, preview_image.height

    extract_boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    stitched_by_channel: dict[int, np.ndarray] = {
        idx: np.zeros((preview_h, preview_w), dtype=np.uint8) for idx, _ in channel_specs
    }

    for extract_index, image_name, bbox in slide_entries:
        npz_path = patient_folder / Path(image_name).with_suffix(".npz")
        x0, y0, x1, y1 = _safe_bbox_to_preview_pixels(bbox, preview_w, preview_h)
        region_h = y1 - y0
        region_w = x1 - x0
        if region_h <= 0 or region_w <= 0:
            continue
        extract_boxes.append((f"{slide_prefix}_{extract_index}", (x0, y0, x1, y1)))

        if not npz_path.exists():
            continue

        npz_data = load_bitpacked_mask_npz(npz_path)
        if "mask" not in npz_data:
            continue

        mask = np.asarray(npz_data["mask"])
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]
        if mask.ndim != 3:
            continue

        for channel_idx, _ in channel_specs:
            if channel_idx >= mask.shape[0]:
                continue
            binary = (mask[channel_idx].astype(np.uint8) * 255)
            tile_image = Image.fromarray(binary, mode="L")
            resized_tile = tile_image.resize((region_w, region_h), resample=Image.Resampling.NEAREST)
            resized_tile_np = np.asarray(resized_tile, dtype=np.uint8)
            stitched_by_channel[channel_idx][y0:y1, x0:x1] = np.maximum(
                stitched_by_channel[channel_idx][y0:y1, x0:x1],
                resized_tile_np,
            )

    images: list[Image.Image] = [preview_image]
    labels: list[str] = [f"WSI Preview: {preview_png_file.name}"]
    for idx, label in channel_specs:
        images.append(Image.fromarray(stitched_by_channel[idx], mode="L").convert("RGB"))
        channel_label = MODEL_OUTPUT_CHANNEL_LABELS[idx] if idx < len(MODEL_OUTPUT_CHANNEL_LABELS) else label
        labels.append(f"Stitched NPZ: {channel_label}")

    app = _ViewerApp(images, labels, max_dim=max_dim, extract_boxes=extract_boxes)
    app.run()
