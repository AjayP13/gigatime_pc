import json
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import torch
from PIL import Image

from utils.device import resolve_device
from utils.model_utils import MODEL_NUM_CLASSES, infer_gigatime_tile, load_gigatime_model
from utils.npz_utils import save_bitpacked_mask_npz

# Trusted pathology slides can exceed Pillow's default decompression bomb limit.
Image.MAX_IMAGE_PIXELS = None

TILE_SIZE = 256
TILE_STRIDE = 128


def _compute_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _build_center_weight(tile_size: int) -> torch.Tensor:
    coords = torch.arange(tile_size, dtype=torch.float32)
    dist = torch.minimum(coords, (tile_size - 1) - coords)
    one_d = (dist + 1.0) / (tile_size / 2.0)
    one_d = torch.clamp(one_d, min=0.05)
    two_d = one_d[:, None] * one_d[None, :]
    return two_d


def _tile_fully_within_any_whitespace(
    tile_bbox: tuple[float, float, float, float], whitespace_bboxes: list[dict[str, Any]]
) -> bool:
    tile_x, tile_y, tile_w, tile_h = tile_bbox
    tile_x2 = tile_x + tile_w
    tile_y2 = tile_y + tile_h
    eps = 1e-8

    for ws in whitespace_bboxes:
        ws_x = float(ws["x_norm"])
        ws_y = float(ws["y_norm"])
        ws_w = float(ws["w_norm"])
        ws_h = float(ws["h_norm"])
        ws_x2 = ws_x + ws_w
        ws_y2 = ws_y + ws_h
        if (
            tile_x >= ws_x - eps
            and tile_y >= ws_y - eps
            and tile_x2 <= ws_x2 + eps
            and tile_y2 <= ws_y2 + eps
        ):
            return True
    return False


def _run_batched_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    batch_size: int,
    image_bbox: tuple[float, float, float, float],
    whitespace_bboxes: list[dict[str, Any]],
    on_batch_processed: Callable[[int], None] | None = None,
) -> torch.Tensor:
    _, image_h, image_w = image_tensor.shape
    y_starts = _compute_starts(image_h, TILE_SIZE, TILE_STRIDE)
    x_starts = _compute_starts(image_w, TILE_SIZE, TILE_STRIDE)

    tile_weight = _build_center_weight(TILE_SIZE)
    accum = torch.zeros((MODEL_NUM_CLASSES, image_h, image_w), dtype=torch.float32)
    weight_accum = torch.zeros((image_h, image_w), dtype=torch.float32)

    pending_tiles: list[torch.Tensor] = []
    pending_coords: list[tuple[int, int, int, int]] = []

    def process_batch(
        batch_tiles: list[torch.Tensor], batch_coords: list[tuple[int, int, int, int]]
    ) -> None:
        if not batch_tiles:
            return
        batch = torch.stack(batch_tiles, dim=0)
        pred_mask = infer_gigatime_tile(model, batch).to(torch.float32)

        for idx, (y0, x0, valid_h, valid_w) in enumerate(batch_coords):
            y1 = y0 + valid_h
            x1 = x0 + valid_w
            local_weight = tile_weight[:valid_h, :valid_w]
            accum[:, y0:y1, x0:x1] += pred_mask[idx, :, :valid_h, :valid_w] * local_weight
            weight_accum[y0:y1, x0:x1] += local_weight
        if on_batch_processed is not None:
            on_batch_processed(1)

    def flush_full_batches() -> None:
        # Keep all in-loop inference calls at exactly `batch_size`.
        while len(pending_tiles) >= batch_size:
            batch_tiles = pending_tiles[:batch_size]
            batch_coords = pending_coords[:batch_size]
            del pending_tiles[:batch_size]
            del pending_coords[:batch_size]
            process_batch(batch_tiles, batch_coords)

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + TILE_SIZE, image_h)
            x1 = min(x0 + TILE_SIZE, image_w)
            valid_h = y1 - y0
            valid_w = x1 - x0

            image_x_norm, image_y_norm, image_w_norm, image_h_norm = image_bbox
            tile_bbox = (
                image_x_norm + (x0 / image_w) * image_w_norm,
                image_y_norm + (y0 / image_h) * image_h_norm,
                (valid_w / image_w) * image_w_norm,
                (valid_h / image_h) * image_h_norm,
            )
            if _tile_fully_within_any_whitespace(tile_bbox, whitespace_bboxes):
                continue

            tile = torch.zeros((3, TILE_SIZE, TILE_SIZE), dtype=torch.float32)
            tile[:, :valid_h, :valid_w] = image_tensor[:, y0:y1, x0:x1]

            pending_tiles.append(tile)
            pending_coords.append((y0, x0, valid_h, valid_w))
            flush_full_batches()

    # Final tail batch (if any) is the only non-full batch by design.
    process_batch(pending_tiles, pending_coords)

    safe_den = torch.clamp(weight_accum, min=1e-6).unsqueeze(0)
    merged = accum / safe_den
    return merged >= 0.5


def _estimate_non_whitespace_tiles(
    image_h: int,
    image_w: int,
    image_bbox: tuple[float, float, float, float],
    whitespace_bboxes: list[dict[str, Any]],
    batch_size: int,
) -> tuple[int, int, int]:
    y_starts = _compute_starts(image_h, TILE_SIZE, TILE_STRIDE)
    x_starts = _compute_starts(image_w, TILE_SIZE, TILE_STRIDE)
    total_tiles = len(y_starts) * len(x_starts)
    non_whitespace_tiles = 0

    image_x_norm, image_y_norm, image_w_norm, image_h_norm = image_bbox
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + TILE_SIZE, image_h)
            x1 = min(x0 + TILE_SIZE, image_w)
            valid_h = y1 - y0
            valid_w = x1 - x0

            tile_bbox = (
                image_x_norm + (x0 / image_w) * image_w_norm,
                image_y_norm + (y0 / image_h) * image_h_norm,
                (valid_w / image_w) * image_w_norm,
                (valid_h / image_h) * image_h_norm,
            )
            if not _tile_fully_within_any_whitespace(tile_bbox, whitespace_bboxes):
                non_whitespace_tiles += 1

    total_batches = (non_whitespace_tiles + batch_size - 1) // batch_size
    return non_whitespace_tiles, total_tiles, total_batches


@click.command("process-patient")
@click.argument("patient_folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--repo-id",
    default="prov-gigatime/GigaTIME",
    show_default=True,
    help="Hugging Face repo id for model weights.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size for processing.",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    show_default=True,
    help="Device to run on: 'auto', 'cpu', 'cuda', or 'mps'.",
)
def process_patient(
    patient_folder: Path,
    repo_id: str,
    batch_size: int,
    device: str,
) -> None:
    """Load the GigaTIME model prior to patient processing."""
    model = load_gigatime_model(repo_id=repo_id)
    torch_device, dtype = resolve_device(device)
    model = model.to(torch_device, dtype=dtype)

    click.echo(
        f"Loaded GigaTIME model 'gigatime' from '{repo_id}' "
        f"-> device={torch_device.type} dtype={dtype} batch_size={batch_size}"
    )

    bbox_path = patient_folder / "bounding_boxes.json"
    if not bbox_path.exists():
        raise click.ClickException(f"Missing required file: {bbox_path}")
    with bbox_path.open("r", encoding="utf-8") as f:
        bbox_index: dict[str, dict[str, Any]] = json.load(f)

    patient_files = sorted(
        path
        for path in patient_folder.rglob("*.png")
        if not path.name.endswith("_preview.png")
    )

    if not patient_files:
        click.echo(
            f"No input files found in '{patient_folder}' (excluding '*_preview.png')."
        )
        return

    click.echo(
        f"Found {len(patient_files)} input file(s) in '{patient_folder}' "
        "(excluding '*_preview.png')."
    )
    patient_id = patient_folder.name
    processable_files = [path for path in patient_files if path.name in bbox_index]
    skipped_files = len(patient_files) - len(processable_files)
    if skipped_files > 0:
        click.echo(
            f"Skipping {skipped_files} file(s): no bounding box entry in bounding_boxes.json"
        )

    for image_idx, file_path in enumerate(processable_files, start=1):

        bbox = bbox_index[file_path.name]
        x_norm = float(bbox["x_norm"])
        y_norm = float(bbox["y_norm"])
        w_norm = float(bbox["w_norm"])
        h_norm = float(bbox["h_norm"])
        whitespace_bboxes = bbox.get("white_space_bboxes", [])

        click.echo(
            f"Processing patient_id={patient_id} image {image_idx}/{len(processable_files)}: "
            f"{file_path.name}"
        )

        with Image.open(file_path) as img:
            rgb = img.convert("RGB")
            image_np = np.asarray(rgb, dtype=np.float32) / 255.0

        image_h, image_w = int(image_np.shape[0]), int(image_np.shape[1])
        non_ws_tiles, total_tiles, total_batches = _estimate_non_whitespace_tiles(
            image_h=image_h,
            image_w=image_w,
            image_bbox=(x_norm, y_norm, w_norm, h_norm),
            whitespace_bboxes=whitespace_bboxes,
            batch_size=batch_size,
        )
        click.echo(
            f"Estimated non-whitespace tiles: {non_ws_tiles}/{total_tiles} "
            f"-> expected batches: {total_batches}"
        )

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        if total_batches > 0:
            with click.progressbar(
                length=total_batches, label=f"Batch progress for {file_path.name}"
            ) as bar:
                merged_mask = _run_batched_inference(
                    model=model,
                    image_tensor=image_tensor,
                    batch_size=batch_size,
                    image_bbox=(x_norm, y_norm, w_norm, h_norm),
                    whitespace_bboxes=whitespace_bboxes,
                    on_batch_processed=bar.update,
                )
        else:
            merged_mask = _run_batched_inference(
                model=model,
                image_tensor=image_tensor,
                batch_size=batch_size,
                image_bbox=(x_norm, y_norm, w_norm, h_norm),
                whitespace_bboxes=whitespace_bboxes,
            )

        out_path = file_path.with_suffix(".npz")
        save_bitpacked_mask_npz(
            out_path,
            merged_mask.numpy(),
            bbox=np.array([x_norm, y_norm, w_norm, h_norm], dtype=np.float32),
            source_image=str(file_path.name),
        )
        click.echo(f"Saved {out_path.name}")

