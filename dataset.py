import bisect
import math
import random
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch.utils.data import Dataset


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


class CarlaDataset(Dataset):
    def __init__(
        self,
        path: str = "../carla-dataset",
        frame_stack: int = 4,
        gaze_temporal_decay: float = 0.8,
        val_split: float = 0.2,
        set_name: Literal["train", "val"] = "train",
    ):
        super().__init__()

        self.path = Path(path)
        self.frame_stack = frame_stack

        layers = int(math.ceil(math.log(0.005, gaze_temporal_decay)))

        self.episodes = []
        self.cum_size = []

        cum_frames = 0
        for route_path in self.path.glob("route_*"):
            if not route_path.is_dir():
                continue

            for seed_path in route_path.glob("seed_*"):
                if not seed_path.is_dir():
                    continue

                obs = torch.load(
                    seed_path / "observations.pt",
                    weights_only=False,
                )
                obs = torch.from_numpy(obs).float()
                obs = pad(obs, frame_stack)

                gaze = torch.load(
                    seed_path / "gaze.pt", map_location="cpu", weights_only=False
                )
                gaze = torch.tensor(gaze)
                gaze = gaze[:, :2]
                gaze = pad(gaze, layers)
                gaze = stack(gaze, layers)
                gaze = pad(gaze, frame_stack)

                actions = torch.load(
                    seed_path / "actions.pt", map_location="cpu", weights_only=False
                )["actions"]

                total_frames = actions.shape[0]
                split_idx = int(total_frames * (1 - val_split))
                if set_name == "val":
                    obs = obs[split_idx:]
                    gaze = gaze[split_idx:]
                    actions = actions[split_idx:]
                    n_frames = total_frames - split_idx
                else:
                    obs = obs[: split_idx + frame_stack - 1]
                    gaze = gaze[: split_idx + frame_stack - 1]
                    actions = actions[:split_idx]
                    n_frames = split_idx

                cum_frames += n_frames
                self.cum_size.append(cum_frames)

                self.episodes.append((obs, gaze, actions))

        self.length = cum_frames

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx = bisect.bisect_right(self.cum_size, idx)
        if ep_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cum_size[ep_idx - 1]

        obs_seq, gaze_seq, action_seq = self.episodes[ep_idx]
        obs_window = obs_seq[local_idx : local_idx + self.frame_stack]
        gaze_window = gaze_seq[local_idx : local_idx + self.frame_stack]
        action = action_seq[local_idx]

        return obs_window, gaze_window, action
