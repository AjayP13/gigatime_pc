import os

import torch
from huggingface_hub import snapshot_download

from utils import archs

MODEL_ARCH = "gigatime"
MODEL_INPUT_CHANNELS = 3
MODEL_NUM_CLASSES = 23
MODEL_OUTPUT_CHANNEL_LABELS = (
    "DAPI",
    "TRITC",  # background channel not used in analysis
    "Cy5",  # background channel not used in analysis
    "PD-1",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
)


def load_gigatime_model(
    repo_id: str = "prov-gigatime/GigaTIME",
    weights_name: str = "model.pth",
    map_location: str = "cpu",
) -> torch.nn.Module:
    """
    Build and load the GigaTIME model from a Hugging Face snapshot.

    Uses the canonical GigaTIME architecture and dimensions:
    - arch='gigatime'
    - input_channels=3
    - num_classes=23
    Output channels map to MODEL_OUTPUT_CHANNEL_LABELS in index order.
    Always constructs and loads weights on CPU first (map_location='cpu').
    """
    model = archs.__dict__[MODEL_ARCH](MODEL_NUM_CLASSES, MODEL_INPUT_CHANNELS)

    local_dir = snapshot_download(repo_id=repo_id)
    weights_path = os.path.join(local_dir, weights_name)

    state_dict = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def infer_gigatime_tile(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Run inference on a single tile (or batched tiles) sized [B,3,256,256].

    - Puts model in eval mode
    - Uses no_grad to avoid autograd overhead
    - Moves/casts input to model's device/dtype
    - Applies sigmoid+threshold to get a binary mask
    - Returns boolean mask tensor of shape [B,23,256,256] on CPU
    """
    if input_image.ndim != 4:
        raise ValueError(f"Expected input of shape [B,3,256,256], got ndim={input_image.ndim}")
    _, c, h, w = input_image.shape
    if c != 3 or h != 256 or w != 256:
        raise ValueError(f"Expected input of shape [B,3,256,256], got {tuple(input_image.shape)}")

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    prepared = input_image.to(device=model_device, dtype=model_dtype, non_blocking=True)

    model.eval()
    use_autocast = model_device.type in ("cuda", "mps") and model_dtype in (torch.float16, torch.bfloat16)
    with torch.no_grad():
        if use_autocast and model_device.type == "cuda":
            autocast_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                logits = model(prepared)
        else:
            # MPS autocast is implicit with dtype=fp16 tensors; CPU runs in fp32 by default.
            logits = model(prepared)

    if logits.ndim != 4:
        raise RuntimeError(f"Model output must be [B,23,256,256], got ndim={logits.ndim}")
    _, out_c, out_h, out_w = logits.shape
    if out_c != MODEL_NUM_CLASSES or out_h != 256 or out_w != 256:
        raise RuntimeError(f"Model output must be [B,23,256,256], got {tuple(logits.shape)}")
    # Convert logits to probabilities and threshold to a boolean mask
    probs = torch.sigmoid(logits)
    mask = probs > threshold
    # Ensure mask is on CPU for downstream use/saving
    return mask.to("cpu")
