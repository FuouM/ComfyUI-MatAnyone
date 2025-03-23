from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from comfy.utils import ProgressBar

from .src.core.inference_core import InferenceCore
from .src.model.matanyone import MatAnyone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = Path(__file__).resolve().parent
cfg = OmegaConf.load(f"{base_dir}/src/base.yaml")


def get_matanyone_model(ckpt_path, device=None) -> MatAnyone:
    # Load the network weights
    cfg["weights"] = ckpt_path
    if device is not None:
        matanyone = MatAnyone(model_cfg=cfg, single_object=True).to(device).eval()
        model_weights = torch.load(ckpt_path, map_location=device)
    else:
        matanyone = MatAnyone(model_cfg=cfg, single_object=True).cuda().eval()
        model_weights = torch.load(ckpt_path)

    matanyone.load_weights(model_weights)

    return matanyone


class MatAnyoneCfg:
    def __init__(self, n_warmup=10) -> None:
        self.n_warmup = n_warmup
        """Number of warmup iterations for the first frame alpha prediction."""


def get_repeat(vframes, index: int, n_warmup: int):
    return vframes[index].unsqueeze(0).repeat(n_warmup, 1, 1, 1)


def preprocess_mask(mask):
    return mask * 255.0


def inference_matanyone(
    vframes: torch.Tensor,
    mask: torch.Tensor,
    processor: InferenceCore,
    n_warmup=10,
):
    mask = mask * 255.0  # from 0..1 to 0..255

    # repeat the first frame for warmup
    repeated_frames = get_repeat(vframes, 0, n_warmup)
    # vframes = torch.cat([repeated_frames, vframes], dim=0).float()
    # length += n_warmup  # update length

    vframes = vframes[1:].to(device)
    length = vframes.shape[0]

    mask = mask.to(device)
    repeated_frames = repeated_frames.to(device)

    # inference start
    phas = []
    pbar = ProgressBar(length)

    for ti in tqdm(range(n_warmup), desc="Warming up"):
        image = repeated_frames[ti]
        if ti == 0:
            # encode given mask
            output_prob = processor.step(image, mask, objects=[1])
            # first frame for prediction
            output_prob = processor.step(image, first_frame_pred=True)
        else:
            # reinit as the first frame for prediction
            output_prob = processor.step(image, first_frame_pred=True)

    # convert output probabilities to alpha matte
    # Get the last warm up frame
    mask = processor.output_prob_to_mask(output_prob)
    phas.append((mask).unsqueeze(0))

    # Process actual frames
    for ti in tqdm(range(length), desc="Process frames"):
        image = vframes[ti]
        output_prob = processor.step(image)
        # convert output probabilities to alpha matte
        mask = processor.output_prob_to_mask(output_prob)
        phas.append((mask).unsqueeze(0))
        pbar.update_absolute(ti, length)

    return phas
