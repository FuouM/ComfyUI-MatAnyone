import torch


def get_mask(
    foreground_mask: torch.Tensor | None = None,
    foreground_MASK: torch.Tensor | None = None,
):
    if foreground_mask is None and foreground_MASK is None:
        raise ValueError("Please provide one mask image")

    if foreground_MASK is not None:
        mask = foreground_MASK.squeeze()
    else:
        mask = img_to_mask(foreground_mask.permute(0, 3, 1, 2)).squeeze()
    return mask


def img_to_mask(tensor: torch.Tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device)
    weights = weights.view(1, 3, 1, 1)
    grayscale = torch.sum(tensor * weights, dim=1, keepdim=True)
    return grayscale


def get_screen(batch_size: int, height: int, width: int, r: int, g: int, b: int):
    # Normalize RGB values to the range 0.0-1.0
    r_normalized = float(r) / 255.0
    g_normalized = float(g) / 255.0
    b_normalized = float(b) / 255.0

    rgb_image = torch.zeros((batch_size, height, width, 3), dtype=torch.float32)
    rgb_image[:, :, :, 0] = r_normalized  # Red channel (index 0)
    rgb_image[:, :, :, 1] = g_normalized  # Green channel (index 1)
    rgb_image[:, :, :, 2] = b_normalized  # Blue channel (index 2)
    return rgb_image
