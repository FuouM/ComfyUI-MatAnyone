"""
@author: Fuou Marinas
@title: FM Nodes
@nickname: FM_nodes
@description: A collection of nodes.
"""

from pathlib import Path

from .utils import get_mask, get_screen

base_dir = Path(__file__).resolve().parent
import torch

# inference_matanyone_extended,
from .constants import ckpt_path
from .mat_anyone import (
    get_matanyone_model,
    inference_matanyone,
)
from .src.core.inference_core import InferenceCore


class MatAnyoneVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_video": ("IMAGE",),
                "mask_frame": (
                    "INT",
                    {"default": 0, "min": 0, "step": 1},
                ),
                "n_warmup": (
                    "INT",
                    {"default": 10, "min": 1, "step": 1},
                ),
            },
            "optional": {
                "foreground_mask": ("IMAGE",),
                "foreground_MASK": ("MASK",),
                "solid_color": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "matte",
        "green_screen",
    )
    FUNCTION = "todo"
    CATEGORY = "MatAnyone"

    def todo(
        self,
        src_video: torch.Tensor,
        mask_frame: int,
        n_warmup: int,
        foreground_mask: torch.Tensor | None = None,
        foreground_MASK: torch.Tensor | None = None,
        solid_color: torch.Tensor | None = None,
    ):
        mask = get_mask(foreground_mask, foreground_MASK)
        src_video = src_video.permute(0, 3, 1, 2)  # T CHW RGB

        # load MatAnyone model
        matanyone = get_matanyone_model(f"{base_dir}/{ckpt_path}")
        processor = InferenceCore(matanyone, cfg=matanyone.cfg)
        phas = inference_matanyone(src_video, mask, processor, mask_frame, n_warmup)
        out_mask = torch.cat(phas).unsqueeze(1).permute(0, 2, 3, 1)

        out_mask_rgb = out_mask.repeat(1, 1, 1, 3)  # Repeat the last dimension 3 times
        gb = torch.empty(
            0, out_mask_rgb.shape[1], out_mask_rgb.shape[2], out_mask_rgb.shape[3]
        )
        if solid_color is not None:
            # T H W C
            src_video_hwc = src_video.permute(0, 2, 3, 1).to(out_mask_rgb.device)
            # x H W C -> T H W C , repeat batch to match video frames
            solid_color_batched = solid_color.repeat(
                src_video_hwc.shape[0], 1, 1, 1
            ).to(out_mask_rgb.device)
            gb = out_mask_rgb * src_video_hwc + (1 - out_mask_rgb) * solid_color_batched

        return (
            out_mask_rgb,
            gb,
        )


class SolidColorBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 720, "min": 1, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 1280, "min": 1, "step": 1},
                ),
                "red": (
                    "INT",
                    {"default": 0, "min": 0, "step": 1},
                ),
                "green": (
                    "INT",
                    {"default": 255, "min": 0, "step": 1},
                ),
                "blue": (
                    "INT",
                    {"default": 0, "min": 0, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("solid",)
    FUNCTION = "todo"
    CATEGORY = "MatAnyone"

    def todo(
        self, batch_size: int, height: int, width: int, red: int, green: int, blue: int
    ):
        return (get_screen(batch_size, height, width, red, green, blue),)
