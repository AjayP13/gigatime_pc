from __future__ import annotations

import re
from pathlib import Path

import click
import matplotlib.pyplot as plt

from commands.view_whole_slide import _ViewerApp, build_whole_slide_viewer


def _slide_sort_key(preview_path: Path) -> tuple[int, str]:
    stem = preview_path.stem
    if stem.endswith("_preview"):
        stem = stem[: -len("_preview")]
    match = re.fullmatch(r"\d+", stem)
    if match is not None:
        return int(stem), preview_path.name
    return 10**12, preview_path.name


@click.command("view-patient")
@click.argument("patient_folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("channels", required=False, default="DAPI")
@click.option(
    "--max-dim",
    type=int,
    default=5000,
    show_default=True,
    help="Maximum size (pixels) of the larger side for each image before layout scaling.",
)
def view_patient(patient_folder: Path, channels: str, max_dim: int) -> None:
    """
    Open all slide preview windows for a patient folder.

    Each window is equivalent to running view-whole-slide on one
    "<slide>_preview.png" file in the patient folder.
    """
    if max_dim <= 0:
        raise click.ClickException("--max-dim must be > 0")

    preview_png_files = sorted(patient_folder.glob("*_preview.png"), key=_slide_sort_key)
    if not preview_png_files:
        raise click.ClickException(f"No '*_preview.png' files found in: {patient_folder}")

    apps: list[_ViewerApp] = [
    ]
    skipped_count = 0
    for preview_png_file in preview_png_files:
        try:
            app = build_whole_slide_viewer(preview_png_file, channels=channels, max_dim=max_dim)
            apps.append(app)
        except click.ClickException as exc:
            if "Could not determine channel count because no NPZ file exists for slide" in str(exc):
                skipped_count += 1
                click.echo(f"Skipping {preview_png_file.name}: no NPZ tiles found for this slide.")
                continue
            raise

    if not apps:
        raise click.ClickException(
            "No slides could be opened. All previews were skipped because matching NPZ tiles were missing."
        )

    for app in apps:
        app.render()

    if skipped_count > 0:
        click.echo(f"Skipped {skipped_count} slide(s) with missing NPZ tiles.")

    plt.show()
