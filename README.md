# ComfyUI-MatAnyone

MatAnyone in ComfyUI (Remove background)

Stable Video Matting with Consistent Memory Propagation: <https://github.com/pq-yang/MatAnyone>

Download `matanyone.pth` from <https://github.com/pq-yang/MatAnyone?tab=readme-ov-file#download-model>

and place it in `checkpoint/matanyone.pth`

## Workflow

[workflow/workflow_mat_anyone.json](workflow/workflow_mat_anyone.json)

![matanyone_example](workflow/mat_anyone_preview.png)
(Not a workflow-embedded image)

Inputs:

- src_video
- `forground_mask` (`IMAGE`) or `foreground_MASK` (`MASK`): The input mask. IMAGE option will automatically convert a black/white image to a mask. At least one option must be given.
- `solid_color` (optional): The solid color to create a screen. Defaults to Green Screen.
- `mask_frame`: The input mask's index (defaults to 0). Support first (0), last and middle frame.
- `n_warmup`: Number of iterations to warm up the model. Defaults to 10.

Your input mask won't actually be in the final matte. Instead, the warmup process generate a new input mask, which is then propagated throughout the video.

## Credit

```cite
@InProceedings{yang2025matanyone,
    title     = {{MatAnyone}: Stable Video Matting with Consistent Memory Propagation},
    author    = {Yang, Peiqing and Zhou, Shangchen and Zhao, Jixin and Tao, Qingyi and Loy, Chen Change},
    booktitle = {arXiv preprint arXiv:2501.14677},
    year      = {2025}
}
```
