from __future__ import annotations

import math
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from PIL import Image
from tkinter import Tk, filedialog

from utils.model_utils import MODEL_OUTPUT_CHANNEL_LABELS
from utils.npz_utils import load_bitpacked_mask_npz


Image.MAX_IMAGE_PIXELS = None


def _resolve_channel_indices(mask_channels: int, channels_csv: str) -> list[tuple[int, str]]:
    requested = [part.strip() for part in channels_csv.split(",") if part.strip()]
    if not requested:
        requested = ["DAPI"]

    labels_upper = {name.upper(): idx for idx, name in enumerate(MODEL_OUTPUT_CHANNEL_LABELS)}
    resolved: list[tuple[int, str]] = []
    for token in requested:
        if token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= mask_channels:
                raise click.ClickException(
                    f"Channel index {idx} is out of bounds for mask with {mask_channels} channels."
                )
            label = (
                MODEL_OUTPUT_CHANNEL_LABELS[idx]
                if idx < len(MODEL_OUTPUT_CHANNEL_LABELS)
                else f"channel_{idx}"
            )
            resolved.append((idx, label))
            continue

        upper = token.upper()
        if upper not in labels_upper:
            supported = ", ".join(MODEL_OUTPUT_CHANNEL_LABELS)
            raise click.ClickException(
                f"Unknown channel '{token}'. Supported names: {supported}"
            )

        idx = labels_upper[upper]
        if idx >= mask_channels:
            raise click.ClickException(
                f"Channel '{token}' maps to index {idx}, but NPZ only has {mask_channels} channels."
            )
        resolved.append((idx, MODEL_OUTPUT_CHANNEL_LABELS[idx]))
    return resolved


def _fit_size(width: int, height: int, max_dim: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 1, 1
    scale = min(max_dim / float(max(width, height)), 1.0)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return out_w, out_h


class _ViewerApp:
    def __init__(self, pil_images: list[Image.Image], labels: list[str], max_dim: int) -> None:
        self.pil_images = [im.convert("RGB") for im in pil_images]
        self.labels = labels
        self.max_dim = max_dim
        self.fig: plt.Figure | None = None

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

    def _render(self) -> None:
        arrays = self._get_resized_rgb_arrays()
        n_items = len(arrays)
        rows, cols = self._compute_grid(n_items)
        fig_w = max(8.0, cols * 3.2)
        fig_h = max(6.0, rows * 3.0 + 0.8)
        self.fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        self.fig.canvas.manager.set_window_title("GigaTIME Image Viewer")
        self.fig.patch.set_facecolor("#1f1f1f")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12, wspace=0.08, hspace=0.22)

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

    def save_screenshot(self) -> None:
        if self.fig is None:
            return

        picker_root = Tk()
        picker_root.withdraw()
        try:
            target = filedialog.asksaveasfilename(
                title="Save screenshot",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
            )
        finally:
            picker_root.destroy()

        if target:
            self.fig.savefig(Path(target), dpi=200, bbox_inches="tight")

    def run(self) -> None:
        self._render()
        if self.fig is None:
            return

        btn_ax = self.fig.add_axes([0.84, 0.02, 0.13, 0.06])
        button = Button(btn_ax, "Screenshot")
        button.on_clicked(lambda _event: self.save_screenshot())
        plt.show()


@click.command("view-image")
@click.argument("png_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("channels", required=False, default="DAPI")
@click.option(
    "--max-dim",
    type=int,
    default=500,
    show_default=True,
    help="Maximum size (pixels) of the larger side for each image before layout scaling.",
)
def view_image(png_file: Path, channels: str, max_dim: int) -> None:
    """
    Open a viewer showing the source PNG and selected NPZ binary maps.

    PNG and NPZ are associated by file stem in the same directory.
    """
    if png_file.suffix.lower() != ".png":
        raise click.ClickException(f"Expected a .png file, got: {png_file}")
    if max_dim <= 0:
        raise click.ClickException("--max-dim must be > 0")

    npz_file = png_file.with_suffix(".npz")
    if not npz_file.exists():
        raise click.ClickException(f"Associated NPZ file not found: {npz_file}")

    with Image.open(png_file) as img:
        png_image = img.convert("RGB")

    npz_data = load_bitpacked_mask_npz(npz_file)
    if "mask" not in npz_data:
        raise click.ClickException(f"Missing 'mask' in NPZ file: {npz_file}")

    mask = np.asarray(npz_data["mask"])
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]
    if mask.ndim != 3:
        raise click.ClickException(f"Expected mask shape [C,H,W] or [H,W], got {mask.shape}")

    channel_specs = _resolve_channel_indices(mask.shape[0], channels)

    images: list[Image.Image] = [png_image]
    labels: list[str] = [f"PNG: {png_file.name}"]
    for idx, label in channel_specs:
        binary = (mask[idx].astype(np.uint8) * 255)
        images.append(Image.fromarray(binary, mode="L").convert("RGB"))
        labels.append(f"NPZ: {label}")

    app = _ViewerApp(images, labels, max_dim=max_dim)
    app.run()
