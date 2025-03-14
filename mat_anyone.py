import numpy as np
from omegaconf import OmegaConf
import torch
from .src.core.inference_core import InferenceCore
from tqdm import tqdm

from .src.model.matanyone import MatAnyone
from pathlib import Path
from comfy.utils import ProgressBar

bgr = (np.array([120, 255, 155], dtype=np.float32) / 255).reshape((1, 1, 3))
# green screen to paste fgr
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


def inference_matanyone(
    vframes: torch.Tensor,
    mask: torch.Tensor,
    processor: InferenceCore,
    n_warmup=10,
):
    length = vframes.shape[0]
    objects = [1]

    # repeat the first frame for warmup
    repeated_frames = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1)
    vframes = torch.cat([repeated_frames, vframes], dim=0).float()
    length += n_warmup  # update length

    mask = mask.to(device)
    vframes = vframes.to(device)
    mask = mask * 255.0

    # inference start
    phas = []
    pbar = ProgressBar(length)
    for ti in tqdm(range(length), desc="Process frames"):
        image = vframes[ti]

        if ti == 0:
            # encode given mask
            output_prob = processor.step(image, mask, objects=objects)
            # first frame for prediction
            output_prob = processor.step(image, first_frame_pred=True)
        else:
            if ti <= n_warmup:
                # reinit as the first frame for prediction
                output_prob = processor.step(image, first_frame_pred=True)
            else:
                output_prob = processor.step(image)

        # convert output probabilities to alpha matte
        mask = processor.output_prob_to_mask(output_prob)

        # DONOT save the warmup frame
        if ti > (n_warmup - 1):
            phas.append((mask).unsqueeze(0))

        pbar.update_absolute(ti, length)

    return phas
