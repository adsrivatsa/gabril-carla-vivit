from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# def gaussian_mask(
#     gaze: torch.Tensor, sigma: float, shape: Tuple[int, int]
# ) -> torch.Tensor:
#     """
#     Generates a single 2D Gaussian mask for a specific gaze point.
#
#     :param gaze: (2). Coordinates of gaze point (normalized 0-1).
#     :param sigma: Standard deviation (spread) of the Gaussian.
#     :param shape: (height, width) of the output mask.
#     :return: torch.Tensor: A 2D tensor of the specified shape.
#     """
#     h, w = shape
#     device = gaze.device
#
#     x0, y0 = gaze
#
#     x0 = x0 * w
#     y0 = y0 * h
#
#     x_grid = torch.arange(0, w, dtype=torch.float32, device=device)
#     y_grid = torch.arange(0, h, dtype=torch.float32, device=device)
#     x, y = torch.meshgrid(x_grid, y_grid, indexing="xy")
#
#     dist_sq = (x - x0) ** 2 + (y - y0) ** 2
#
#     mask = torch.exp(-dist_sq / (2.0 * sigma**2))
#
#     return mask
import torch


def decaying_gaussian_mask(
    gaze_coords: torch.Tensor,
    shape: Tuple[int, int],
    base_sigma: float = 5.0,  # Gamma in formula
    temporal_decay: float = 0.9,  # Alpha in formula
    blur_growth: float = 0.96,  # Beta in formula
) -> torch.Tensor:
    """
    Generates cumulative heatmaps with Recasens et al. formulation:
    Sum of Gaussians with fading amplitude and growing variance.

    Formula: Sum[ alpha^|j| * N(x, gamma * beta^-|j|) ]

    :param gaze_coords: (..., layers, 2). Last dim is (x, y), 2nd-to-last is Time.
                        Values should be normalized coordinates [0, 1].
    :param shape: (height, width) of the image.
    :param base_sigma: The size of the spot at the current frame (T=0).
    :param temporal_decay: How fast intensity fades (0.0 to 1.0).
    :param blur_growth: How fast variance grows/blurs backwards (0.0 to 1.0).
                        (Note: <1.0 means it grows as you go back in time).
    :return: (..., height, width). Batch dims matching input.
    """
    H, W = shape
    device = gaze_coords.device

    # 1. Flatten all batch dimensions into one 'B' dimension
    batch_shape = gaze_coords.shape[:-2]
    layers = gaze_coords.shape[-2]

    # Shape becomes (Total_Batch_Size, Layers, 2)
    gaze_flat = gaze_coords.view(-1, layers, 2)
    B, L, _ = gaze_flat.shape

    # 2. Precompute Grid
    x_range = torch.arange(0, W, dtype=torch.float32, device=device)
    y_range = torch.arange(0, H, dtype=torch.float32, device=device)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="xy")

    # Broadcast grid to batch size: (1, H, W)
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    heatmap = torch.zeros((B, H, W), dtype=torch.float32, device=device)

    # We assume the last frame in the stack is T=0 (the target)
    target_idx = L - 1

    for t in range(L):
        # j = distance from target frame (e.g., -5, -4, ... 0)
        j = t - target_idx
        abs_j = abs(j)

        # --- A. Amplitude Decay (Alpha^|j|) ---
        weight = temporal_decay**abs_j

        # --- B. Variance Growth (Gamma * Beta^-|j|) ---
        # Note: Since beta < 1, raising to negative power makes sigma larger
        current_sigma = base_sigma * (blur_growth**-abs_j)
        denom = 2.0 * current_sigma**2

        # --- C. Get Coordinates from FLATTENED tensor ---
        # (Fix: used gaze_flat instead of gaze_coords)
        pt = gaze_flat[:, t, :]

        # Check for NaNs (invalid gaze points)
        valid = ~torch.isnan(pt).any(dim=1).view(B, 1, 1)

        # Scale normalized coords to H, W
        x0 = (pt[:, 0] * W).view(B, 1, 1)
        y0 = (pt[:, 1] * H).view(B, 1, 1)

        # Squared Euclidean distance
        dist_sq = (grid_x - x0) ** 2 + (grid_y - y0) ** 2

        # --- D. Gaussian Calculation ---
        gauss = torch.exp(-dist_sq / denom) * valid

        # --- E. Accumulate (Sum) ---
        heatmap += weight * gauss

    # 3. Normalize per image in the batch
    # We find the max value for *each* image (dim 1 and 2) to normalize 0-1
    max_vals = heatmap.flatten(1).max(dim=1).values.view(B, 1, 1)

    # Avoid division by zero
    heatmap = heatmap / (max_vals + 1e-6)

    return heatmap.view(*batch_shape, H, W)


def patchify(gaze_mask: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
    """
    Splits an image/mask into non-overlapping patches.

    :param gaze_mask: (..., H, W).
    :param patch_size: (Patch_H, Patch_W) in pixels.
    :return: (..., Grid_Rows, Grid_Cols, Patch_H, Patch_W).
    """
    patch_h, patch_w = patch_size

    batch_shape = gaze_mask.shape[:-2]
    H, W = gaze_mask.shape[-2:]

    assert H % patch_h == 0 and W % patch_w == 0, (
        f"Image size ({H},{W}) must be divisible by patch size ({patch_h},{patch_w})"
    )

    num_patches_h = H // patch_h
    num_patches_w = W // patch_w

    gaze_mask = gaze_mask.reshape(-1, H, W)
    B, _, _ = gaze_mask.shape

    gaze_mask = gaze_mask.view(B, num_patches_h, patch_h, num_patches_w, patch_w)

    gaze_mask = gaze_mask.permute(0, 1, 3, 2, 4)

    return gaze_mask.view(*batch_shape, num_patches_h, num_patches_w, patch_h, patch_w)


def plot_patches(patches: torch.Tensor, spacing: int = 1):
    """
    Visualizes patches using PyTorch operations.

    :param patches: (Grid_Rows, Grid_Cols, Patch_H, Patch_W)
    :param spacing: Spacing between patches in pixels.
    """
    if patches.ndim != 4:
        raise ValueError(f"Patches must be 4D, got {patches.shape}")

    patches = patches.detach().cpu()

    rows, cols, ph, pw = patches.shape

    full_h = (rows * ph) + ((rows - 1) * spacing)
    full_w = (cols * pw) + ((cols - 1) * spacing)

    canvas = torch.full((full_h, full_w), float("nan"), dtype=patches.dtype)

    for i in range(rows):
        for j in range(cols):
            y_start = i * (ph + spacing)
            x_start = j * (pw + spacing)

            canvas[y_start : y_start + ph, x_start : x_start + pw] = patches[i, j]

    plt.figure(figsize=(8, 8))

    cmap = plt.cm.hot
    cmap.set_bad(color="grey")

    plt.imshow(canvas.numpy(), cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Layout: {rows}x{cols} | Patch Size: {ph}x{pw}")
    plt.show()


# mask = decaying_gaussian_mask([(0.5, 0.5), (0.4, 0.6), (1, 0.5)], 5, (84, 84), 0.5)
#
# patched_mask = patchify(mask, patch_size=(6, 6))
# plot_patches(patched_mask, spacing=1)
