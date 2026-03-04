import bisect
import math
import os
import random
from pathlib import Path
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# CARLA task routing (route_id, seed)
# -----------------------------------------------------------------------------

TASK_TO_ROUTE = {
    "Mixed_": {
        "train": [
            (r, s)
            for r in [24759, 25857, 24211, 3100, 2416, 3472, 25863, 26408, 27494, 24258]
            for s in range(200, 220)
        ],
        "test": [
            (r, 400)
            for r in sorted(
                [24759, 25857, 24211, 3100, 2416, 3472, 25863, 26408, 27494, 24258]
            )
        ],
        "test_unseen": [
            (r, 400)
            for r in sorted(
                [18305, 1852, 24224, 3099, 3184, 3464, 27529, 26401, 2215, 25951]
            )
        ],
    },
    "ParkingCutIn_": {
        "train": [(24759, s) for s in range(200, 220)],
        "test": [(24759, 400)],
        "test_unseen": [(18305, 400)],
    },
    "AccidentTwoWays_": {
        "train": [(25857, s) for s in range(200, 220)],
        "test": [(25857, 400)],
        "test_unseen": [(1852, 400)],
    },
    "DynamicObjectCrossing_": {
        "train": [(24211, s) for s in range(200, 220)],
        "test": [(24211, 400)],
        "test_unseen": [(24224, 400)],
    },
    "CrossingBicycleFlow_": {
        "train": [(3100, s) for s in range(200, 220)],
        "test": [(3100, 400)],
        "test_unseen": [(3099, 400)],
    },
    "VanillaNonSignalizedTurnEncounterStopsign_": {
        "train": [(2416, s) for s in range(200, 220)],
        "test": [(2416, 400)],
        "test_unseen": [(3184, 400)],
    },
    "VehicleOpensDoorTwoWays_": {
        "train": [(3472, s) for s in range(200, 220)],
        "test": [(3472, 400)],
        "test_unseen": [(3464, 400)],
    },
    "PedestrianCrossing_": {
        "train": [(25863, s) for s in range(200, 220)],
        "test": [(25863, 400)],
        "test_unseen": [(27529, 400)],
    },
    "MergerIntoSlowTrafficV2_": {
        "train": [(26408, s) for s in range(200, 220)],
        "test": [(26408, 400)],
        "test_unseen": [(26401, 400)],
    },
    "BlockedIntersection_": {
        "train": [(27494, s) for s in range(200, 220)],
        "test": [(27494, 400)],
        "test_unseen": [(2215, 400)],
    },
    "HazardAtSideLaneTwoWays_": {
        "train": [(24258, s) for s in range(200, 220)],
        "test": [(24258, 400)],
        "test_unseen": [(25951, 400)],
    },
}

MAX_EPISODES_CARLA = {k: len(v["train"]) for k, v in TASK_TO_ROUTE.items()}

# -----------------------------------------------------------------------------
# decaying_gaussian_mask (Recasens et al. formulation for "mine" loading)
# -----------------------------------------------------------------------------


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
    Supports both past and future gaze via symmetric temporal decay and blur.

    Formula: Sum[ alpha^|j| * N(x, gamma * beta^-|j|) ]
    where j is the distance from the target (current) frame in either direction.

    :param gaze_coords: (..., layers, 2). Last dim is (x, y), 2nd-to-last is Time.
                        Center index is the current frame (T=0). Values [0, 1].
    :param shape: (height, width) of the image.
    :param base_sigma: The size of the spot at the current frame (T=0).
    :param temporal_decay: How fast intensity fades (0.0 to 1.0) with distance.
    :param blur_growth: How fast variance grows/blurs with distance (0.0 to 1.0).
                        (Note: <1.0 means it grows as you go away from T=0).
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

    # Center of window is T=0 (the target frame); applies to past and future
    target_idx = (L - 1) // 2

    for t in range(L):
        # j = distance from target frame (negative = past, positive = future)
        j = t - target_idx
        abs_j = abs(j)

        # --- A. Amplitude Decay (Alpha^|j|) ---
        weight = temporal_decay**abs_j

        # --- B. Variance Growth (Gamma * Beta^-|j|) ---
        # Note: Since beta < 1, raising to negative power makes sigma larger
        current_sigma = base_sigma * (blur_growth**-abs_j)
        denom = 2.0 * current_sigma**2

        # --- C. Get Coordinates from FLATTENED tensor ---
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


# -----------------------------------------------------------------------------
# GazeToMask (gabril-style: precomputed Gaussians with sigma/coefficient decay)
# -----------------------------------------------------------------------------


class GazeToMask:
    """
    Generates saliency masks from gaze coordinates using precomputed Gaussians
    with temporal decay (sigma and coefficient per layer). Follows gabril Atari style.
    """

    def __init__(
        self,
        height: int,
        width: int,
        sigmas: List[float],
        coeficients: List[float],
    ):
        assert len(sigmas) == len(coeficients)
        self.height = height
        self.width = width
        self.sigmas = sigmas
        self.coeficients = coeficients

    def _gaussian_at(
        self, mean_x: float, mean_y: float, sigma: float, coef: float
    ) -> torch.Tensor:
        """Single Gaussian centered at (mean_x*W, mean_y*H), scaled by coef."""
        x = (
            torch.arange(self.width, dtype=torch.float32)
            .view(1, -1)
            .expand(self.height, self.width)
        )
        y = (
            torch.arange(self.height, dtype=torch.float32)
            .view(-1, 1)
            .expand(self.height, self.width)
        )
        x0, y0 = mean_x * self.width, mean_y * self.height
        g = coef * torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
        return g

    def find_bunch_of_maps(
        self, means: List[List[float]], offset_start: int = 0
    ) -> torch.Tensor:
        """
        Sum Gaussians centered at each mean.
        :param means: List of [x, y] normalized coords in [0, 1].
        :param offset_start: Index into sigmas/coeficients for the first mean.
        """
        result = torch.zeros(self.height, self.width)
        for i, (mx, my) in enumerate(means):
            idx = offset_start + i
            if idx >= len(self.sigmas):
                break
            sigma = self.sigmas[idx]
            coef = self.coeficients[idx]
            result = result + self._gaussian_at(mx, my, sigma, coef)
        return result / (result.max() + 1e-8)


def gabril_gaze_windows(
    episode_gaze: List[torch.Tensor],
    short_memory_length: int = 20,
    stride: int = 2,
) -> List[torch.Tensor]:
    """
    Build per-timestep gaze coordinate windows for gabril-style saliency.
    Each window is (ep_len, num_sigmas, 2) with NaN for missing entries.

    :param episode_gaze: List of (ep_len, gaze_dim) per episode.
    :param short_memory_length: Past/future frames in window.
    :param stride: Subsampling stride for gaze.
    """
    num_sigmas = 2 * short_memory_length + 1
    windowed = []
    for ep in episode_gaze:
        ep_coords = ep[:, :2]
        L = len(ep_coords)
        windows = torch.full((L, num_sigmas, 2), float("nan"))

        for j in range(L):
            start = max(0, j - stride * short_memory_length)
            end_idx = min(short_memory_length * stride + j + 1, L)
            offset_start = max(short_memory_length - j, 0)
            coords = ep_coords[start:end_idx:stride]
            k = len(coords)
            windows[j, offset_start : offset_start + k] = coords

        windowed.append(windows)
    return windowed


def gabril_gaze_mask_carla(
    gaze_windows: torch.Tensor,
    height: int = 180,
    width: int = 320,
    gaze_sigma: float = 30.0,
    gaze_alpha: float = 0.8,
    gaze_beta: float = 0.99,
) -> torch.Tensor:
    """
    Compute saliency masks from gaze windows (gabril-style GazeToMask).

    :param gaze_windows: (..., 41, 2) with NaN for missing coords.
    :param height: Output mask height.
    :param width: Output mask width.
    :return: (..., height, width) saliency masks.
    """
    short_memory_length = 20
    saliency_sigmas = [
        gaze_sigma / (gaze_beta ** (short_memory_length - i))
        for i in range(short_memory_length + 1)
    ]
    coeficients = [
        gaze_alpha ** (short_memory_length - i) for i in range(short_memory_length + 1)
    ]
    coeficients = coeficients + coeficients[::-1][1:]
    saliency_sigmas = saliency_sigmas + saliency_sigmas[::-1][1:]

    masker = GazeToMask(height, width, saliency_sigmas, coeficients)

    batch_shape = gaze_windows.shape[:-2]
    num_sigmas = gaze_windows.shape[-2]
    flat = gaze_windows.reshape(-1, num_sigmas, 2)
    B = flat.shape[0]

    results = torch.zeros(B, height, width)
    for b in range(B):
        valid = ~torch.isnan(flat[b, :, 0])
        indices = torch.where(valid)[0]
        if len(indices) == 0:
            continue
        means = flat[b, indices].tolist()
        offset_start = indices[0].item()
        results[b] = masker.find_bunch_of_maps(means=means, offset_start=offset_start)

    return results.reshape(*batch_shape, height, width)


# -----------------------------------------------------------------------------
# gabril_load_data_carla: load CARLA data following gabril_load_data pattern
# -----------------------------------------------------------------------------

CARLA_HEIGHT = 180
CARLA_WIDTH = 320


def _load_episode(
    datapath: str,
    route_id: int,
    seed: int,
    frame_stack: int,
    grayscale: bool,
    use_gaze: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a single CARLA episode.
    :return: episode_obs (N, F, C, H, W), actions (N,), gaze_windows (N, F, 41, 2), gaze_coords (N, 2)
    """
    path = Path(datapath) / f"route_{route_id}" / f"seed_{seed}"

    obs = torch.load(path / "observations.pt", weights_only=False)
    obs = torch.from_numpy(obs).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    actions = torch.load(path / "actions.pt", weights_only=False)["actions"]
    actions = torch.from_numpy(np.stack(actions))

    gaze_info = None
    if use_gaze:
        gaze_info = torch.tensor(torch.load(path / "gaze.pt", weights_only=False))
        gaze_info = gaze_info[:, :2].clamp(0.0, 1.0)

    # Frame stacking: prepend (frame_stack-1) copies of first frame
    obs = torch.cat([obs[0:1].expand(frame_stack - 1, -1, -1, -1), obs], dim=0)

    n_actions = len(actions)
    if use_gaze and gaze_info is not None:
        episode_gaze = [gaze_info]
        gaze_windows = gabril_gaze_windows(episode_gaze)[0]  # (N, 41, 2)
        gaze_coords = gaze_info  # (N, 2)
        # Pad gaze_windows for stacking: first (frame_stack-1) rows repeat frame 0
        gaze_windows = torch.cat(
            [gaze_windows[0:1].expand(frame_stack - 1, -1, -1), gaze_windows],
            dim=0,
        )
    else:
        num_sigmas = 41
        gaze_windows = torch.full((len(obs), num_sigmas, 2), float("nan"))
        gaze_coords = torch.zeros(n_actions, 2)

    # Stack: for each t, window is [t-frame_stack+1 : t+1]
    new_obs, new_gw = [], []
    for s in range(frame_stack):
        end = None if s == frame_stack - 1 else s - frame_stack + 1
        new_obs.append(obs[s:end])
        new_gw.append(gaze_windows[s:end])

    episode_obs = torch.stack(new_obs, dim=1)  # (N, F, C, H, W)
    episode_gaze_windows = torch.stack(new_gw, dim=1)  # (N, F, 41, 2)

    if grayscale:
        episode_obs = episode_obs.float().mean(dim=2, keepdim=True)

    return episode_obs, actions, episode_gaze_windows, gaze_coords


def gabril_load_data_carla(
    task: str,
    datapath: str,
    frame_stack: int = 4,
    grayscale: bool = False,
    num_episodes: int | None = None,
    use_gaze: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load CARLA data following gabril_load_data pattern.
    Returns gaze_windows (not precomputed masks) for use with gabril_gaze_mask_carla.

    :param task: Key in TASK_TO_ROUTE (e.g. 'Mixed_', 'ParkingCutIn_').
    :param datapath: Base path containing route_X/seed_Y/.
    :return: observations (B, F, C, H, W), actions (B), gaze_windows (B, F, 41, 2), gaze_coords (B, 2)
    """
    routes = TASK_TO_ROUTE[task]["train"]
    max_ep = MAX_EPISODES_CARLA[task]
    n_ep = num_episodes if num_episodes is not None else max_ep
    assert n_ep <= max_ep, (
        f"Requested {n_ep} episodes but only {max_ep} available for {task}"
    )

    episodes_obs = []
    episodes_actions = []
    episodes_gaze_windows = []
    episodes_gaze_coords = []

    for route_id, seed in tqdm(routes[:n_ep], desc="Loading CARLA episodes"):
        obs, acts, gw, gc = _load_episode(
            datapath, route_id, seed, frame_stack, grayscale, use_gaze
        )
        episodes_obs.append(obs)
        episodes_actions.append(acts)
        episodes_gaze_windows.append(gw)
        episodes_gaze_coords.append(gc)

    observations = torch.cat(episodes_obs, dim=0)
    actions = torch.cat(episodes_actions, dim=0)
    gaze_windows = torch.cat(episodes_gaze_windows, dim=0)
    gaze_coords = torch.cat(episodes_gaze_coords, dim=0)

    observations = observations.float() / 255.0
    actions = actions.long()

    return observations, actions, gaze_windows, gaze_coords


def plot_gaze_and_obs(
    gaze: torch.Tensor,
    obs: torch.Tensor,
    save_path: str | None = None,
) -> None:
    """
    Plot gaze mask, observation, and their element-wise product.
    :param gaze: (H, W) or (C, H, W)
    :param obs: (H, W) or (C, H, W), uint8 or float
    """
    y1 = gaze.detach().cpu().float()
    y2 = obs.detach().cpu().float()
    if obs.dtype == torch.uint8:
        y2 = y2 / 255.0

    y3 = (y1 * y2).float()

    if y3.ndim == 3:
        if y3.shape[0] == 3:
            y3 = y3.permute(1, 2, 0)
        elif y3.shape[0] == 1:
            y3 = y3.squeeze(0)
    if y1.ndim == 3:
        y1 = y1.squeeze(0) if y1.shape[0] == 1 else y1.permute(1, 2, 0)
    if y2.ndim == 3:
        y2 = y2.squeeze(0) if y2.shape[0] == 1 else y2.permute(1, 2, 0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(y1.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("Gaze")
    ax2.imshow(y2.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    ax2.set_title("Observation")
    ax3.imshow(y3.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    ax3.set_title("Gaze × Obs")
    for ax in (ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def pad(t: torch.Tensor, k: int) -> torch.Tensor:
    first_slice = t[0:1]
    padding = first_slice.expand(k - 1, *([-1] * (t.ndim - 1)))
    padded = torch.cat([padding, t], dim=0)
    return padded


def stack(t: torch.Tensor, k: int) -> torch.Tensor:
    windows = t.unfold(dimension=0, size=k, step=1)
    ndim = windows.ndim
    permute_order = (0, ndim - 1) + tuple(range(1, ndim - 1))
    stacked = windows.permute(permute_order)
    return stacked


def list_games(path: str) -> List[str]:
    return [entry.name for entry in os.scandir(path) if entry.is_dir()]


def load_data(
    folder: str, device: str, gaze_temporal_decay: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load Carla data for training and testing.

    :param folder: The folder of the Carla dataset.
    :param device: The device to use.
    :param gaze_temporal_decay: The gaze temporal decay, used to calculate
        the amount of stacking layers (until opacity = 5%).
    :return: (B, F, C, H, W), (B, F, layers, 2), (B)
    """
    observations = []
    gazes = []
    actions = []

    layers = int(math.ceil(math.log(0.005, gaze_temporal_decay)))

    seed_list = [p for p in Path(folder).iterdir() if p.is_dir()]
    for seed in seed_list:
        seed_observations = torch.from_numpy(
            torch.load(seed / "observations.pt", weights_only=False)
        ).to(dtype=torch.float)
        seed_observations = pad(seed_observations, 4)
        seed_observations = stack(seed_observations, 4)
        observations.append(seed_observations)

        seed_gaze = torch.tensor(torch.load(seed / "gaze.pt", weights_only=False)).to(
            dtype=torch.float
        )
        seed_gaze = pad(seed_gaze, layers)
        seed_gaze = stack(seed_gaze, layers)
        seed_gaze = pad(seed_gaze, 4)
        seed_gaze = stack(seed_gaze, 4)
        gazes.append(seed_gaze)

        actions.append(
            torch.from_numpy(
                torch.load(seed / "actions.pt", weights_only=False)["actions"]
            ).to(dtype=torch.float)
        )

    observations = torch.cat(observations, dim=0).to(device=device)
    gazes = torch.cat(gazes, dim=0).to(device=device)
    actions = torch.cat(actions, dim=0).to(device=device)

    observations = observations / 255.0

    gazes = gazes[:, :, :, :2]
    gazes = torch.clamp(gazes, 0, 1)

    return observations.permute(0, 1, 4, 2, 3), gazes, actions
