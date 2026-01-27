import math
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import augmentation
import gaze
import wandb
from dataset import CarlaDataset
from device import device
from vivit import FactorizedViViTV1


@dataclass
class Config:
    # paths and flags
    game_index: int = 0
    carla_dataset_folder: str = "./carla-dataset"
    use_plots: bool = False
    save_folder: str = "./models"
    version: int = 2
    seed: int = 42

    # gaze
    gaze_sigma: int = 30
    gaze_beta: float = 0.99
    gaze_alpha: float = 0.8

    # augmentation
    augment_shift_pad: int = 6
    augment_noise_std: float = 0.01

    # transformer arch
    spatial_patch_size: Tuple[int, int] = (16, 9)
    embedding_dim: int = 128
    spatial_depth: int = 3
    temporal_depth: int = 2
    spatial_heads: int = 4
    temporal_heads: int = 4
    inner_dim: int = 32
    mlp_dim: int = 256
    dropout: float = 0.1

    # hyperparams
    learning_rate: float = 5e-4
    epochs: int = 1000
    train_pct: float = 0.8
    batch_size: int = 32
    lambda_gaze: float = 1
    weight_decay: float = 1e-2
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    clip_grad_norm: float = 1.0
    warmup_epochs: int = 10
    warmup_start_factor: float = 1e-10
    min_learning_rate: float = 1e-6

    # testing
    test_episodes: int = 10
    max_episode_length: int = 5000


# def test_agent(args: Config, model: torch.nn.Module) -> float:
#     """
#     Runs the model in the actual Gym environment to measure performance.
#     """
#     env_name = GYM_ENV_MAP.get(args.game)
#     if env_name is None:
#         print(f"Warning: No Gym environment found for '{args.game}' in GYM_ENV_MAP.")
#         return 0.0
#
#     env = gym.make(env_name, render_mode="rgb_array")
#     env = ResizeObservation(env, (84, 84))
#     env = GrayscaleObservation(env, keep_dim=False)
#     env = FrameStackObservation(env, 4)
#
#     action_meanings = env.unwrapped.get_action_meanings()
#     fire_a = -1
#     if "FIRE" in action_meanings:
#         fire_a = action_meanings.index("FIRE")
#
#     total_reward = 0
#
#     model.eval()
#     for i in range(args.test_episodes):
#         obs, _ = env.reset()
#         done = False
#         ep_reward = 0
#         steps = 0
#
#         if fire_a != -1:
#             obs, _, _, _, _ = env.step(fire_a)
#
#         while not done and steps < args.max_episode_length:
#             steps += 1
#
#             obs = torch.from_numpy(obs).float() / 255.0
#             F, H, W = obs.shape
#             obs = obs.view(1, F, 1, H, W).to(device=device)
#
#             with torch.no_grad():
#                 pred_a, _ = model(obs)
#                 action = pred_a.argmax(dim=1).item()
#
#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             ep_reward += reward
#
#         total_reward += ep_reward
#
#     env.close()
#
#     return total_reward / args.test_episodes


def save_checkpoint(
    path: str,
    epoch: int,
    best_reward: float,
    wandb_id: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: SequentialLR = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    scheduler_state = None
    if scheduler is not None:
        scheduler_state = scheduler.state_dict()

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "scaler_state_dict": scaler.state_dict(),
        "best_reward": best_reward,
        "wandb_id": wandb_id,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(checkpoint_data, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: SequentialLR = None,
) -> Tuple[int, float, str | None]:
    if not os.path.exists(path):
        return 0, -float("inf"), None

    print(f"--> Found checkpoint! Resuming from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNGs if available
    if "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        torch.set_rng_state(rng["torch"].cpu())
        cuda_states = [state.cpu() for state in rng["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    # Extract metadata
    start_epoch = checkpoint["epoch"] + 1
    best_reward = checkpoint.get("best_reward", -float("inf"))
    wandb_id = checkpoint.get("wandb_id", None)

    print(f"--> Resumed at Epoch {start_epoch}, Best Reward: {best_reward:.4f}")
    return start_epoch, best_reward, wandb_id


def calculate_loss(
    args: Config,
    model: torch.nn.Module,
    obs: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
):
    with autocast(device_type="cuda", dtype=torch.float16):
        pred_a, cls_attn = model(obs)

        # behavior cloning loss
        policy_loss = Fn.cross_entropy(pred_a, a)

        # gaze loss
        _, _, GH, GW = g.shape

        cls_attn = cls_attn.mean(dim=2)  # (B, F, T)
        _, F, T = cls_attn.shape
        cls_attn = cls_attn.view(
            -1, F, GH // args.spatial_patch_size[0], GW // args.spatial_patch_size[1]
        )

        cls_attn = Fn.interpolate(
            cls_attn,
            size=(GH, GW),
            mode="bilinear",
            align_corners=False,
        )

        gaze_loss = torch.norm(cls_attn - g, p="fro", dim=(1, 2)) ** 2
        gaze_loss = gaze_loss.mean()

        return pred_a, policy_loss, gaze_loss


def train(
    args: Config,
):
    """
    Train a ViViT model.

    :param args: Config.
    :return:
    """
    resume_path = f"{args.save_folder}/latest_checkpoint.pt"

    # B, F, C, H, W = observations.shape
    # n_actions = torch.max(actions).item() + 1
    #
    # all_actions = actions.view(-1).long()
    # class_counts = torch.bincount(all_actions)
    # num_classes = len(class_counts)
    #
    # safe_counts = class_counts.float()
    # safe_counts[safe_counts == 0] = 1.0
    #
    # total_samples = len(all_actions)
    # class_weights = total_samples / (num_classes * safe_counts)
    # class_weights = torch.sqrt(class_weights)
    # class_weights = torch.clamp(class_weights, min=1.0, max=10.0)
    # class_weights = class_weights.to(device=device)

    model = FactorizedViViTV1(
        image_size=(180, 320),
        patch_size=args.spatial_patch_size,
        frames=4,
        channels=3,
        action_space_dim=7,
        dim=args.embedding_dim,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        spatial_heads=args.spatial_heads,
        temporal_heads=args.temporal_heads,
        dim_head=args.inner_dim,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        use_flash_attn=True,
        return_cls_attn=True,
        use_temporal_mask=True,
    ).to(device=device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = GradScaler()

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )

    decay_epochs = args.epochs - args.warmup_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=decay_epochs, eta_min=args.min_learning_rate
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    start_epoch, best_reward, wandb_id = load_checkpoint(
        resume_path, model, optimizer, scaler, scheduler
    )

    if wandb_id is None:
        wandb_id = wandb.util.generate_id()

    group_id = (
        f"v{args.version}_"
        f"lr{args.learning_rate:.0e}_"
        f"lam{args.lambda_gaze}_"
        f"dim{args.embedding_dim}_"
        f"pt{args.spatial_patch_size[0]}_"
        f"d{args.dropout}"
    )
    # run = wandb.init(
    #     entity="papaya147-ml",
    #     project="GABRIL-Carla-ViViT",
    #     config=args.__dict__,
    #     group=group_id,
    #     name=f"v{args.version}",
    #     job_type="train",
    #     id=wandb_id,
    #     resume="allow",
    # )

    train_dataset = CarlaDataset(
        path=args.carla_dataset_folder,
        gaze_temporal_decay=args.gaze_beta,
        val_split=0.2,
        set_name="train",
    )
    input("ready")
    exit()
    val_dataset = CarlaDataset(
        path=args.carla_dataset_folder,
        gaze_temporal_decay=args.gaze_beta,
        val_split=0.2,
        set_name="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    for e in range(start_epoch, args.epochs):
        metrics = {
            "train_loss": 0,
            "train_policy_loss": 0,
            "train_gaze_loss": 0,
            "train_acc": 0,
            "val_loss": 0,
            "val_policy_loss": 0,
            "val_gaze_loss": 0,
            "val_acc": 0,
        }

        # train loop
        model.train()
        for obs, g, a in train_loader:
            obs, g = preprocess(args, obs, g)
            print(obs.shape, g.shape)
            exit()
            a = a.to(device=device)

            optimizer.zero_grad()

            pred_a, policy_loss, gaze_loss = calculate_loss(model, obs, g, a)
            loss = policy_loss + args.lambda_gaze * gaze_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            acc = (pred_a.argmax(dim=1) == a).float().sum()

            curr_batch_size = obs.size(0)

            metrics["train_loss"] += loss.item() * curr_batch_size
            metrics["train_policy_loss"] += policy_loss.item() * curr_batch_size
            metrics["train_gaze_loss"] += gaze_loss.item() * curr_batch_size
            metrics["train_acc"] += acc.item()

        # validation
        model.eval()
        with torch.no_grad():
            for obs, g, a in val_loader:
                obs, g = preprocess(args, obs, g, augment=False)
                a = a.to(device=device)

                pred_a, policy_loss, gaze_loss = calculate_loss(model, obs, g, a)
                loss = policy_loss + args.lambda_gaze * gaze_loss

                acc = (pred_a.argmax(dim=1) == a).float().sum()

                curr_batch_size = obs.size(0)

                metrics["val_loss"] += loss.item() * curr_batch_size
                metrics["val_policy_loss"] += policy_loss.item() * curr_batch_size
                metrics["val_gaze_loss"] += gaze_loss.item() * curr_batch_size
                metrics["val_acc"] += acc.item()

        scheduler.step()

        # testing
        mean_reward = test_agent(args, model)

        log_data = {
            k: v / train_size if "train" in k else v / val_size
            for k, v in metrics.items()
        }
        log_data["epoch"] = e
        log_data["reward"] = mean_reward
        log_data["learning_rate"] = optimizer.param_groups[0]["lr"]

        run.log(data=log_data)

        if mean_reward > best_reward:
            best_reward = mean_reward
            save_path = f"{args.save_folder}/{args.game}/best_reward.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        save_checkpoint(
            resume_path, e, best_reward, wandb_id, model, optimizer, scaler, scheduler
        )

    save_path = f"{args.save_folder}/{args.game}/final.pt"
    torch.save(model.state_dict(), save_path)

    run.finish()


def preprocess(
    args: Config,
    observations: torch.Tensor,
    gaze_coords: torch.Tensor,
    augment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the observations and gaze masks. Convert the gaze masks into patches.
    Normalize the gaze patches.

    :param args: Config.
    :param observations: (B, F, C, H, W)
    :param gaze_coords: (B, F, layers, 2)
    :param augment: Augment the data with random shifts and noise?
    :return: (B, F, C, H, W), (B, F, T)
    """
    B, F, C, H, W = observations.shape
    random_example = random.randint(0, len(observations) - 1)

    gaze_masks = gaze.decaying_gaussian_mask(
        gaze_coords,
        shape=(H, W),
        base_sigma=args.gaze_sigma,
        temporal_decay=args.gaze_alpha,
        blur_growth=args.gaze_beta,
    )

    aug_observations = observations.to(device=device)
    aug_gaze_masks = gaze_masks.to(device=device)
    if augment:
        aug_observations, aug_gaze_masks = augmentation.random_shift(
            aug_observations, aug_gaze_masks, pad=args.augment_shift_pad
        )
        aug_observations, aug_gaze_masks = augmentation.random_noise(
            aug_observations, aug_gaze_masks, std=args.augment_noise_std
        )

    # # plotting random observations and gazes
    # if args.use_plots:
    #     atari.plot_frames(aug_observations[random_example])
    #     atari.plot_frames(aug_gaze_masks.unsqueeze(2)[random_example])

    # gaze_mask_patches = gaze.patchify(
    #     aug_gaze_masks, patch_size=args.spatial_patch_size
    # )
    # B, F, gridR, gridR, patchR, patchC = gaze_mask_patches.shape
    #
    # # plotting random gaze patches
    # if args.use_plots:
    #     gaze.plot_patches(gaze_mask_patches[random_example][0], 1)
    #
    # # pooling the last 2 dims of gaze
    # gaze_mask_patches = gaze_mask_patches.mean(dim=(-2, -1))
    # gaze_mask_patches = gaze_mask_patches.view(B, F, gridR * gridR)
    #
    # normalizing
    gaze_sums = aug_gaze_masks.sum(dim=(-2, -1), keepdim=True)
    aug_gaze_masks = aug_gaze_masks / (gaze_sums + 1e-8)

    return aug_observations, aug_gaze_masks  # gaze_mask_patches


def set_seed(seed: int):
    """
    Sets the seed for all sources of randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    args = Config()

    set_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
