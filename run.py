"""
@author: Fuou Marinas
@title: FM Nodes
@nickname: FM_nodes
@description: A collection of nodes.
"""

from pathlib import Path

base_dir = Path(__file__).resolve().parent
import torch

from .src.core.inference_core import InferenceCore
from .mat_anyone import get_matanyone_model, inference_matanyone
from .constants import ckpt_path


def img_to_mask(tensor: torch.Tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device)
    weights = weights.view(1, 3, 1, 1)
    grayscale = torch.sum(tensor * weights, dim=1, keepdim=True)
    return grayscale


class MatAnyoneVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_video": ("IMAGE",),
                "n_warmup": (
                    "INT",
                    {"default": 10, "min": 0, "step": 1},
                ),
            },
            "optional": {"foreground_mask": ("IMAGE",), "foreground_MASK": ("MASK",)},
        }

    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "matte",
    )
    FUNCTION = "todo"
    CATEGORY = "MatAnyone"

    def todo(
        self,
        src_video: torch.Tensor,
        n_warmup: int,
        foreground_mask: torch.Tensor | None = None,
        foreground_MASK: torch.Tensor | None = None,
    ):
        if foreground_mask is None and foreground_MASK is None:
            raise ValueError("Please provide one mask image")

        if foreground_MASK is not None:
            mask = foreground_MASK.squeeze()
        else:
            mask = img_to_mask(foreground_mask.permute(0, 3, 1, 2)).squeeze()
        src_video = src_video.permute(0, 3, 1, 2)  # T CHW RGB

        # load MatAnyone model
        matanyone = get_matanyone_model(f"{base_dir}/{ckpt_path}")
        processor = InferenceCore(matanyone, cfg=matanyone.cfg)
        phas = inference_matanyone(src_video, mask, processor, n_warmup)
        out_mask = torch.cat(phas).unsqueeze(1).permute(0, 2, 3, 1)
        out_mask_rgb = out_mask.repeat(1, 1, 1, 3)  # Repeat the last dimension 3 times

        return (out_mask_rgb,)
