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


def get_repeat(vframes: torch.Tensor, index: int, n_warmup: int):
    return vframes[index].unsqueeze(0).repeat(n_warmup, 1, 1, 1)


def preprocess_mask(mask):
    return mask * 255.0


def warming_up(mask, processor, n_warmup, repeated_frames):
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
    return output_prob


def inference_matanyone(
    vframes: torch.Tensor,
    mask: torch.Tensor,
    processor: InferenceCore,
    frame_index: int = 0,
    n_warmup=10,
):
    mask = preprocess_mask(mask)
    vframes = vframes.to(device)
    mask = mask.to(device)
    length = vframes.shape[0]
    phas = [None] * length  # initialize output list

    if frame_index >= length:
        raise ValueError(f"Expected 0 <= x < {length}, got {frame_index}")

    if frame_index == 0:
        # Case 1: frame_index == 0 (original behavior)
        repeated_frames = get_repeat(vframes, 0, n_warmup)
        repeated_frames = repeated_frames.to(device)

        output_prob = warming_up(mask, processor, n_warmup, repeated_frames)
        phas[0] = processor.output_prob_to_mask(output_prob).unsqueeze(0)

        pbar = ProgressBar(length - 1)
        for ti in tqdm(range(1, length), desc="Forward Propagation"):
            image = vframes[ti]
            output_prob = processor.step(image)
            phas[ti] = processor.output_prob_to_mask(output_prob).unsqueeze(0)
            pbar.update_absolute(ti - 1, length - 1)

    elif frame_index == length - 1:
        # Case 2: frame_index == last frame (reverse propagation)
        reversed_vframes = torch.flip(vframes, dims=[0])
        # repeat the last frame
        repeated_frames = get_repeat(reversed_vframes, 0, n_warmup)
        repeated_frames = repeated_frames.to(device)
        reversed_mask = mask  # mask for the last frame

        output_prob = warming_up(reversed_mask, processor, n_warmup, repeated_frames)
        phas[frame_index] = processor.output_prob_to_mask(output_prob).unsqueeze(0)

        pbar = ProgressBar(length - 1)
        for ti in tqdm(range(1, length), desc="Backward Propagation"):
            image = reversed_vframes[ti]
            output_prob = processor.step(image)
            phas[length - 1 - ti] = processor.output_prob_to_mask(
                output_prob
            ).unsqueeze(0)  # reverse index
            pbar.update_absolute(ti - 1, length - 1)

    elif 0 < frame_index < length - 1:
        # Case 3: 0 < frame_index < last frame (forward and backward)
        # repeat the frame_index frame
        repeated_frames = get_repeat(vframes, frame_index, n_warmup)
        repeated_frames = repeated_frames.to(device)

        # Warm up at frame_index
        output_prob = warming_up(mask, processor, n_warmup, repeated_frames)
        phas[frame_index] = processor.output_prob_to_mask(output_prob).unsqueeze(0)

        # Forward Propagation (from frame_index + 1 to end)
        pbar_forward = ProgressBar(length - 1 - frame_index)
        for ti in tqdm(range(frame_index + 1, length), desc="Forward Propagation"):
            image = vframes[ti]
            output_prob = processor.step(image)
            phas[ti] = processor.output_prob_to_mask(output_prob).unsqueeze(0)
            pbar_forward.update_absolute(
                ti - (frame_index + 1), length - 1 - frame_index
            )

        # Backward Propagation (from frame_index - 1 to start)
        # frames before frame_index, reversed
        reversed_vframes_backward = torch.flip(vframes[:frame_index], dims=[0])
        pbar_backward = ProgressBar(frame_index)
        for ti in tqdm(range(frame_index), desc="Backward Propagation"):
            image = reversed_vframes_backward[ti]
            output_prob = processor.step(image)
            phas[frame_index - 1 - ti] = processor.output_prob_to_mask(
                output_prob
            ).unsqueeze(0)  # reverse index
            pbar_backward.update_absolute(ti, frame_index)

    return phas
