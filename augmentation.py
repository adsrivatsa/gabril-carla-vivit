from typing import Tuple

import torch
import torch.nn.functional as Fn


def random_shift(
    x: torch.Tensor, gaze: torch.Tensor = None, pad: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad by K pixels, then random crop back to the original size.
    Consistent across frames per stack, but random across batches.

    :param x: (B, F, C, H, W)
    :param gaze: (B, F, H, W)
    :param pad: Padding to add before cropping.
    :return: (B, F, C, H, W), (B, F, H, W)
    """
    B, F, C, H, W = x.shape
    device = x.device

    x_reshaped = x.view(B * F, C, H, W)  # (B * F, C, H, W)
    x_padded = Fn.pad(
        x_reshaped, (pad,) * 4, mode="replicate"
    )  # (B * F, C, H + 2 * pad, W + 2 * pad)
    x_padded = x_padded.view(
        B, F, C, H + 2 * pad, W + 2 * pad
    )  # (B, F, C, H + 2 * pad, W + 2 * pad)

    gaze_padded = None
    if gaze is not None:
        gaze_reshaped = gaze.view(B * F, H, W)  # (B * F, H, W)
        gaze_padded = Fn.pad(
            gaze_reshaped, (pad,) * 4, mode="constant", value=0
        )  # (B * F, H + 2 * pad, W +  2* pad)
        gaze_padded = gaze_padded.view(
            B, F, H + 2 * pad, W + 2 * pad
        )  # (B, F, H + 2 * pad, W + 2 * pad)

    crop_x = torch.randint(0, 2 * pad + 1, (B,), device=device)  # (B)
    crop_y = torch.randint(0, 2 * pad + 1, (B,), device=device)  # (B)

    base_y = torch.arange(H, device=device).view(1, H)  # (1, H)
    base_x = torch.arange(W, device=device).view(1, W)  # (1, W)

    grid_y = (base_y + crop_y.view(B, 1)).unsqueeze(2)  # (B, H, 1)
    grid_x = (base_x + crop_x.view(B, 1)).unsqueeze(1)  # (B, 1, W)

    b_idx = torch.arange(B, device=device).view(B, 1, 1)  # (B, 1, 1)

    x_out = x_padded[b_idx, :, :, grid_y, grid_x]  # (B, H, W, F, C)
    x_out = x_out.permute(0, 3, 4, 1, 2)  # (B, F, C, H, W)

    gaze_out = None
    if gaze is not None:
        gaze_out = gaze_padded[b_idx, :, grid_y, grid_x]  # (B, H, W, F)
        gaze_out = gaze_out.permute(0, 3, 1, 2)  # (B, F, H, W)

    return x_out, gaze_out


def random_noise(
    x: torch.Tensor, gaze: torch.Tensor = None, std: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adds Gaussian noise to x.

    If gaze is provided:
        - Areas with Gaze=1.0 get 0% noise (Clear).
        - Areas with Gaze=0.0 get 100% noise (Noisy).

    :param x: (B, F, C, H, W)
    :param gaze: (B, F, H, W)
    :param std: Standard deviation of the Gaussian noise.
    :return: (B, F, C, H, W), (B, F, H, W)
    """
    device = x.device

    noise = torch.randn_like(x, device=device) * std

    if gaze is not None:
        noise_scale = 1.0 - gaze.unsqueeze(2)

        x_noisy = x + (noise * noise_scale)
    else:
        x_noisy = x + noise

    x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

    return x_noisy, gaze
