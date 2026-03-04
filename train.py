import datetime
import math
import os
import random
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

import checkpoint
import dataset
from augmentation import Augment
from config import config
from device import device
from vivit import AuxGazeFactorizedViViT, FactorizedViViT

# TF32 disabled for reproducibility; enable for faster training if needed
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def evaluate_agent(
    model: torch.nn.Module,
    split: Literal["test", "val"],
    save_dir: Optional[str] = None,
):
    """
    Run CARLA rollout evaluation when carla_port and routes_path are set.
    Uses carla/rollout.py for self-contained simulation. Otherwise returns placeholder.
    """
    episodes = config.test_episodes if split == "test" else config.val_episodes

    # CARLA eval disabled (carla_port=None skips rollouts)
    if config.carla_port is None:
        ep_returns = np.zeros(episodes)
        ep_steps = np.zeros(episodes, dtype=int)
        best_rollout_obs = np.zeros((1, 3, 180, 320), dtype=np.uint8)
        best_rollout_g = np.zeros((1, 180, 320), dtype=np.float32)
        best_rollout_overlaid = np.zeros((1, 3, 180, 320), dtype=np.uint8)
        return ep_returns, ep_steps, best_rollout_obs, best_rollout_g, best_rollout_overlaid

    routes_path = config.carla_routes_path
    if not os.path.isabs(routes_path):
        routes_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), routes_path
        )
    if not os.path.exists(routes_path):
        print(f"Routes file not found: {routes_path}")
        ep_returns = np.zeros(episodes)
        ep_steps = np.zeros(episodes, dtype=int)
        best_rollout_obs = np.zeros((1, 3, 180, 320), dtype=np.uint8)
        best_rollout_g = np.zeros((1, 180, 320), dtype=np.float32)
        best_rollout_overlaid = np.zeros((1, 3, 180, 320), dtype=np.uint8)
        return ep_returns, ep_steps, best_rollout_obs, best_rollout_g, best_rollout_overlaid

    route_ids = [
        str(r) for r, _ in dataset.TASK_TO_ROUTE[config.task]["test"]
    ]

    if not route_ids:
        ep_returns = np.zeros(episodes)
        ep_steps = np.zeros(episodes, dtype=int)
        best_rollout_obs = np.zeros((1, 3, 180, 320), dtype=np.uint8)
        best_rollout_g = np.zeros((1, 180, 320), dtype=np.float32)
        best_rollout_overlaid = np.zeros((1, 3, 180, 320), dtype=np.uint8)
        return ep_returns, ep_steps, best_rollout_obs, best_rollout_g, best_rollout_overlaid

    from rollout import run_single_rollout

    H, W = 180, 320
    ep_returns = np.zeros(episodes)
    ep_steps = np.zeros(episodes, dtype=int)
    best_rollout_obs = np.zeros((1, 3, H, W), dtype=np.uint8)
    best_rollout_g = np.zeros((1, H, W), dtype=np.float32)
    best_rollout_overlaid = np.zeros((1, 3, H, W), dtype=np.uint8)
    best_score = -1.0

    for ep in range(min(episodes, len(route_ids))):
        route_id = route_ids[ep % len(route_ids)]
        seed = config.seed + ep
        result = run_single_rollout(
            model=model,
            routes_file=routes_path,
            route_id=route_id,
            host=config.carla_host,
            port=config.carla_port,
            traffic_manager_port=config.carla_traffic_manager_port,
            seed=seed,
            frame_rate=20.0,
            obs_res=(H, W),
            frame_stack=config.frame_stack,
            act_dim=7,
            patch_size=config.spatial_patch_size,
            max_steps=config.max_episode_length,
            waypoint_threshold=3.0,
            noop_steps=10,
        )
        score = result.get("score_composed", 0.0)
        ep_returns[ep] = score
        ep_steps[ep] = result.get("steps", 0)
        if "error" in result:
            print(f"Rollout {ep} (route {route_id}): {result['error']}")

        if score > best_score and result.get("obs_frames"):
            best_score = score
            obs_frames = result["obs_frames"]
            gaze_frames = result.get("gaze_frames", [])
            n = min(len(obs_frames), len(gaze_frames), 30)
            if n > 0:
                obs = np.stack(obs_frames[:n], axis=0)
                obs = np.transpose(obs, (0, 3, 1, 2))
                best_rollout_obs = obs[:1]
                if gaze_frames:
                    g = np.stack(gaze_frames[:n], axis=0)
                    g_min, g_max = g.min(), g.max()
                    g = (g - g_min) / (g_max - g_min + 1e-8)
                    best_rollout_g = g[:1].astype(np.float32)
                    overlaid = obs[:1].astype(np.float32) * np.expand_dims(
                        g[:1], 1
                    )
                    best_rollout_overlaid = np.clip(
                        overlaid, 0, 255
                    ).astype(np.uint8)

    return ep_returns, ep_steps, best_rollout_obs, best_rollout_g, best_rollout_overlaid


def gaze_kl_loss(cls_attn: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Compute gaze regularization loss using KL divergence.

    If gaze_loss_mode == "mean_then_kl": average attention across heads first,
    then compute a single KL divergence per (batch, frame).

    If gaze_loss_mode == "kl_then_mean": compute KL divergence per head first,
    then average the per-head losses.

    :param cls_attn: (B, F, SpatialHeads, T) — raw CLS attention from model
    :param g: (B, F, GH, GW) — gaze target distribution (already normalized)
    :return: scalar gaze loss
    """
    _, _, GH, GW = g.shape
    eps = 1e-8

    if config.gaze_loss_mode == "mean_then_kl":
        # average across heads, then KL
        attn = cls_attn.mean(dim=2)  # (B, F, T)
        B, F, _ = attn.shape
        attn = attn.view(
            B,
            F,
            GH // config.spatial_patch_size[0],
            GW // config.spatial_patch_size[1],
        )  # (B, F, PH, PW)
        attn = Fn.interpolate(
            attn,
            size=(GH, GW),
            mode="bilinear",
            align_corners=False,
        )  # (B, F, GH, GW)

        attn_flat = attn.reshape(B * F, -1) + eps
        gaze_flat = g.reshape(B * F, -1) + eps

        attn_flat = attn_flat / attn_flat.sum(dim=1, keepdim=True)
        gaze_flat = gaze_flat / gaze_flat.sum(dim=1, keepdim=True)

        loss = torch.sum(
            gaze_flat * (torch.log(gaze_flat) - torch.log(attn_flat)), dim=1
        )
        return loss.mean()

    elif config.gaze_loss_mode == "kl_then_mean":
        # KL per head, then average across heads
        B, F, heads, T = cls_attn.shape
        pH = GH // config.spatial_patch_size[0]
        pW = GW // config.spatial_patch_size[1]

        # reshape each head's tokens to spatial grid and interpolate
        attn = cls_attn.permute(2, 0, 1, 3)  # (heads, B, F, T)
        attn = attn.reshape(heads * B, F, pH, pW)
        attn = Fn.interpolate(
            attn,
            size=(GH, GW),
            mode="bilinear",
            align_corners=False,
        )  # (heads * B, F, GH, GW)
        attn = attn.reshape(heads, B, F, GH, GW)

        # expand gaze target to match heads dimension
        gaze_exp = g.unsqueeze(0).expand(heads, -1, -1, -1, -1)  # (heads, B, F, GH, GW)

        attn_flat = attn.reshape(heads, B * F, -1) + eps
        gaze_flat = gaze_exp.reshape(heads, B * F, -1) + eps

        attn_flat = attn_flat / attn_flat.sum(dim=2, keepdim=True)
        gaze_flat = gaze_flat / gaze_flat.sum(dim=2, keepdim=True)

        # KL per head: (heads, B*F)
        kl = torch.sum(gaze_flat * (torch.log(gaze_flat) - torch.log(attn_flat)), dim=2)
        # mean over heads, then mean over (batch, frame)
        return kl.mean()

    else:
        raise ValueError(f"Unknown gaze_loss_mode: {config.gaze_loss_mode}")


def calculate_loss(
    model: torch.nn.Module,
    obs: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
):
    with autocast(device_type=device, dtype=torch.float16):
        pred_a, cls_attn = model(
            obs
        )  # pred_a: (B, act_dim), cls_attn: (B, F, SpatialHeads, T)

        # behavior cloning loss (MSE for continuous actions)
        policy_loss = Fn.mse_loss(pred_a, a)

        # gaze loss (skip when no_gaze is set)
        if config.no_gaze:
            gaze_loss = torch.tensor(0.0, device=obs.device)
        else:
            gaze_loss = gaze_kl_loss(cls_attn, g)

        return pred_a, policy_loss, gaze_loss


def train(
    observations: torch.Tensor,
    gaze_coords: torch.Tensor,
    actions: torch.Tensor,
):
    """
    Train a ViViT model.

    :param observations: (B, F, C, H, W)
    :param gaze_coords: Gaze coordinate data — (B, F, layers, 2) for mine,
                         or (B, F, 41, 2) windowed coords for gabril.
                         Masks are computed per-batch in preprocess.
    :param actions: (B, act_dim) for continuous
    :return:
    """
    run_id = config.run_id
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    algo_label = f"{config.algorithm}_NoGaze" if config.no_gaze else config.algorithm
    run_name = f"{algo_label}_{config.task}_seed-{config.seed}_{run_id}_{date_str}"
    save_dir = os.path.join(config.save_folder, run_id)
    resume_path = os.path.join(save_dir, "latest_checkpoint.pt")

    B, F, C, H, W = observations.shape
    # Continuous actions: (B, act_dim)
    actions = actions.float()
    act_dim = actions.shape[-1] if actions.dim() > 1 else 1
    if actions.dim() == 1:
        actions = actions.unsqueeze(-1)  # (B,) -> (B, 1)

    if config.algorithm == "AuxGazeFactorizedViViT":
        model = AuxGazeFactorizedViViT(
            image_size=(H, W),
            patch_size=config.spatial_patch_size,
            frames=F,
            channels=C,
            n_classes=act_dim,
            dim=config.embedding_dim,
            spatial_depth=config.spatial_depth,
            temporal_depth=config.temporal_depth,
            spatial_heads=config.spatial_heads,
            temporal_heads=config.temporal_heads,
            dim_head=config.inner_dim,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            use_flash_attn=True,
            return_cls_attn=True,
            num_registers=config.num_registers,
        )
    else:
        model = FactorizedViViT(
            image_size=(H, W),
            patch_size=config.spatial_patch_size,
            frames=F,
            channels=C,
            n_classes=act_dim,
            dim=config.embedding_dim,
            spatial_depth=config.spatial_depth,
            temporal_depth=config.temporal_depth,
            spatial_heads=config.spatial_heads,
            temporal_heads=config.temporal_heads,
            dim_head=config.inner_dim,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            use_flash_attn=True,
            return_cls_attn=True,
            num_registers=config.num_registers,
        )
    count_model_params(model, verbose=True)
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = GradScaler()

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )

    decay_epochs = config.epochs - config.warmup_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=decay_epochs, eta_min=config.min_learning_rate
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )

    # run = wandb.init(
    #     entity="papaya147-ml",
    #     project="ViViT-GABRIL-CARLA",
    #     config=config.__dict__,
    #     name=run_name,
    #     job_type="train",
    #     id=run_id,
    #     resume="allow",
    # )

    dataset_len = len(observations)
    train_size = int(config.train_pct * dataset_len)
    val_size = dataset_len - train_size

    train_obs, val_obs = observations[:train_size], observations[train_size:]
    train_gaze, val_gaze = (
        gaze_coords[:train_size],
        gaze_coords[train_size:],
    )
    train_acts, val_acts = actions[:train_size], actions[train_size:]

    train_dataset = TensorDataset(train_obs, train_gaze, train_acts)
    val_dataset = TensorDataset(val_obs, val_gaze, val_acts)

    # Deterministic shuffling via generator for reproducibility
    train_generator = torch.Generator().manual_seed(config.seed)
    val_generator = torch.Generator().manual_seed(config.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=val_generator,
    )

    start_epoch, best_return = checkpoint.load(
        resume_path,
        model,
        optimizer,
        scaler,
        scheduler,
        train_generator=train_generator,
        val_generator=val_generator,
    )

    for e in range(start_epoch, config.epochs):
        metrics = {
            "train/train_loss": 0,
            "train/train_policy_loss": 0,
            "train/train_gaze_loss": 0,
            "train/train_mae": 0,
        }

        # train loop
        model.train()
        for obs, g, a in train_loader:
            obs, g = preprocess(obs, g)  # obs: (B, F, C, H, W), g: (B, F, H, W)
            a = a.to(device=device).float()
            if a.dim() == 1:
                a = a.unsqueeze(-1)  # (B,) -> (B, 1)

            optimizer.zero_grad()

            pred_a, policy_loss, gaze_loss = calculate_loss(
                model, obs, g, a
            )  # pred_a: (B, act_dim)
            loss = policy_loss + config.lambda_gaze * gaze_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            mae = Fn.l1_loss(pred_a, a)

            curr_batch_size = obs.size(0)

            metrics["train/train_loss"] += loss.item() * curr_batch_size
            metrics["train/train_policy_loss"] += policy_loss.item() * curr_batch_size
            metrics["train/train_gaze_loss"] += gaze_loss.item() * curr_batch_size
            metrics["train/train_mae"] += mae.item() * curr_batch_size

        # validation (every epoch)
        metrics["eval/val_loss"] = 0
        metrics["eval/val_policy_loss"] = 0
        metrics["eval/val_gaze_loss"] = 0
        metrics["eval/val_mae"] = 0

        model.eval()
        with torch.no_grad():
            for obs, g, a in val_loader:
                obs, g = preprocess(
                    obs, g, augment=False
                )  # obs: (B, F, C, H, W), g: (B, F, H, W)
                a = a.to(device=device).float()
                if a.dim() == 1:
                    a = a.unsqueeze(-1)  # (B,) -> (B, 1)

                pred_a, policy_loss, gaze_loss = calculate_loss(model, obs, g, a)
                loss = policy_loss + config.lambda_gaze * gaze_loss

                mae = Fn.l1_loss(pred_a, a)

                curr_batch_size = obs.size(0)

                metrics["eval/val_loss"] += loss.item() * curr_batch_size
                metrics["eval/val_policy_loss"] += policy_loss.item() * curr_batch_size
                metrics["eval/val_gaze_loss"] += gaze_loss.item() * curr_batch_size
                metrics["eval/val_mae"] += mae.item() * curr_batch_size

        # rollouts (every 100 epochs)
        mean_return = -1
        if (e + 1) % 100 == 0:
            (
                ep_returns,
                ep_steps,
                best_rollout_obs,
                best_rollout_g,
                best_rollout_overlaid,
            ) = evaluate_agent(model=model, split="val", save_dir=save_dir)
            mean_return = float(ep_returns.mean())

            if mean_return > best_return:
                best_return = mean_return
                best_save_path = f"{save_dir}/best_return.pt"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), best_save_path)

        scheduler.step()

        log_data = {
            k: v / train_size if "train" in k else v / val_size
            for k, v in metrics.items()
        }
        log_data["epoch"] = e
        log_data["train/learning_rate"] = optimizer.param_groups[0]["lr"]
        if mean_return != -1:
            log_data["eval/mean_return"] = mean_return
            log_data["eval/std_return"] = float(ep_returns.std())
            log_data["eval/max_return"] = float(ep_returns.max())
            log_data["eval/min_return"] = float(ep_returns.min())

            log_data["eval/mean_steps"] = float(ep_steps.mean())
            log_data["eval/std_steps"] = float(ep_steps.std())
            log_data["eval/best_rollout_obs"] = wandb.Video(
                best_rollout_obs, fps=15, format="gif"
            )
            log_data["eval/best_rollout_g"] = wandb.Video(
                best_rollout_g, fps=15, format="gif"
            )
            log_data["eval/best_rollout_overlaid"] = wandb.Video(
                best_rollout_overlaid, fps=15, format="gif"
            )

        print(log_data)
        run.log(data=log_data)

        checkpoint.save(
            resume_path,
            e,
            best_return,
            model,
            optimizer,
            scaler,
            scheduler,
            train_generator=train_generator,
            val_generator=val_generator,
        )

    # testing and saving final model
    final_save_path = os.path.join(save_dir, "final.pt")
    torch.save(model.state_dict(), final_save_path)

    ep_returns, ep_steps, _, _, _ = evaluate_agent(
        model=model, split="test", save_dir=save_dir
    )

    mean_val = np.mean(ep_returns)
    std_val = np.std(ep_returns)
    max_val = np.max(ep_returns)
    min_val = np.min(ep_returns)

    eval_table = wandb.Table(data=[[r] for r in ep_returns], columns=["return"])

    summary_table = wandb.Table(
        columns=["Run Name", "Mean Return", "Std Dev", "Max Return", "Min Return"],
        data=[
            [
                run.name,
                f"{mean_val:.2f}",
                f"{std_val:.2f}",
                f"{max_val:.2f}",
                f"{min_val:.2f}",
            ]
        ],
    )

    run.log(
        {
            "test/final/returns": eval_table,
            "test/final/return_distribution": wandb.plot.histogram(
                eval_table, "return"
            ),
            "test/final/mean_return": mean_val,
            "test/final/std_return": std_val,
            "test/final/max_return": max_val,
            "test/final/min_return": min_val,
            "test/final/summary_return": summary_table,
        }
    )

    final_model = wandb.Artifact(f"{run.name}-final-model", type="model")
    final_model.add_file(final_save_path)
    run.log_artifact(final_model)

    # testing and saving best model
    best_save_path = os.path.join(save_dir, "best_return.pt")
    model.load_state_dict(
        torch.load(best_save_path, map_location=device, weights_only=False)
    )

    ep_returns, ep_steps, _, _, _ = evaluate_agent(
        model=model, split="test", save_dir=save_dir
    )

    mean_val = np.mean(ep_returns)
    std_val = np.std(ep_returns)
    max_val = np.max(ep_returns)
    min_val = np.min(ep_returns)

    eval_table = wandb.Table(data=[[r] for r in ep_returns], columns=["return"])

    summary_table = wandb.Table(
        columns=["Run Name", "Mean Return", "Std Dev", "Max Return", "Min Return"],
        data=[
            [
                run.name,
                f"{mean_val:.2f}",
                f"{std_val:.2f}",
                f"{max_val:.2f}",
                f"{min_val:.2f}",
            ]
        ],
    )

    run.log(
        {
            "test/best/returns": eval_table,
            "test/best/return_distribution": wandb.plot.histogram(eval_table, "return"),
            "test/best/mean_return": mean_val,
            "test/best/std_return": std_val,
            "test/best/max_return": max_val,
            "test/best/min_return": min_val,
            "test/best/summary_return": summary_table,
        }
    )

    best_model = wandb.Artifact(f"{run.name}-best-model", type="model")
    best_model.add_file(best_save_path)
    run.log_artifact(best_model)

    run.finish()


def preprocess(
    observations: torch.Tensor,
    gaze_coords: torch.Tensor,
    augment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gaze masks from gaze coordinates, augment the observations and
    gaze masks, and normalize the gaze masks.

    :param observations: (B, F, C, H, W)
    :param gaze_coords: Gaze coordinate data — (B, F, layers, 2) for mine,
                         or (B, F, 41, 2) windowed coords for gabril.
    :param augment: Augment the data with random shifts, color jitter and noise?
    :return: (B, F, C, H, W), (B, F, H, W)
    """
    B, F, C, H, W = observations.shape

    if config.loading_method == "mine":
        gaze_masks = dataset.decaying_gaussian_mask(
            gaze_coords=gaze_coords,
            shape=(H, W),
            base_sigma=config.gaze_sigma,
            temporal_decay=config.gaze_alpha,
            blur_growth=config.gaze_beta,
        )
    else:
        gaze_masks = dataset.gabril_gaze_mask_carla(
            gaze_coords,
            height=H,
            width=W,
            gaze_sigma=config.gaze_sigma,
            gaze_alpha=config.gaze_alpha,
            gaze_beta=config.gaze_beta,
        )

    random_example = random.randint(0, len(observations) - 1)

    # pre augmentation plots
    if config.use_plots:
        plot_frames(observations[random_example])
        plot_frames(gaze_masks.unsqueeze(2)[random_example])

    if augment:
        augment_fn = Augment(
            frame_shape=(F, C, H, W),
            crop_padding=config.augment_crop_padding,
            cutout_hole_size=config.augment_cutout_hole_size,
            p_spatial_corruption=config.augment_p_spatial_corruptions,
            seed=config.seed,
        )
        observations, gaze_masks = augment_fn(observations, gaze_masks)

    observations = observations.to(device=device)  # (B, F, C, H, W)
    gaze_masks = gaze_masks.to(device=device)  # (B, F, H, W)

    # post augmentation plots
    if config.use_plots:
        plot_frames(observations[random_example])
        plot_frames(gaze_masks.unsqueeze(2)[random_example])

    # normalizing
    gaze_sums = gaze_masks.sum(dim=(-2, -1), keepdim=True)
    gaze_masks = gaze_masks / (gaze_sums + 1e-8)  # (B, F, H, W)

    return observations, gaze_masks


def plot_frames(frames: torch.Tensor):
    """
    Plots frames from a (F, C, H, W) tensor in a square grid.

    :param frames: (F, C, H, W). Values should be roughly in [0, 1] for floats or [0, 255] for uint8.
    """
    frames = frames.detach().cpu().numpy()

    if frames.ndim != 4:
        raise ValueError(f"Expected shape (F, C, H, W), got {frames.shape}")

    F, C, H, W = frames.shape

    cols = math.ceil(math.sqrt(F))
    rows = math.ceil(F / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]

        if i < F:
            img = frames[i]

            img = np.transpose(img, (1, 2, 0))

            if C == 1:
                ax.imshow(img.squeeze(-1))
            else:
                ax.imshow(img)

            ax.set_title(f"Frame {i}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def count_model_params(model: torch.nn.Module, verbose: bool = True) -> int:
    """
    Count and optionally print model parameters.

    :param model: PyTorch model
    :param verbose: If True, print total params and per-module breakdown
    :return: Total number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")
        for name, module in model.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                print(f"  {name}: {n:,}")

    return trainable


def set_seed(seed: int):
    """
    Sets the seed for all sources of randomness to ensure reproducible training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuDNN: deterministic algorithms (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Disable TF32 for bit-exact reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    set_seed(config.seed)

    print(f"Task: {config.task}")

    if config.loading_method == "mine":
        # For mine: folder must contain seed_* subdirs (e.g. route_X with seed_200, seed_201, ...)
        # Use first route from task's train routes
        route_id = dataset.TASK_TO_ROUTE[config.task]["train"][0][0]
        folder = f"{config.carla_dataset_folder}/route_{route_id}"
        observations, gaze_coords, actions = dataset.load_data(
            folder, device="cpu", gaze_temporal_decay=config.gaze_alpha
        )
        actions = actions.float()
    elif config.loading_method == "gabril":
        observations, actions, gaze_coords, _ = dataset.gabril_load_data_carla(
            task=config.task,
            datapath=config.carla_dataset_folder,
            frame_stack=config.frame_stack,
            num_episodes=dataset.MAX_EPISODES_CARLA[config.task],
        )
        actions = actions.float()  # continuous actions
        # gaze_coords here is gaze_windows (B, F, 41, 2) for gabril_gaze_mask_carla
    else:
        raise ValueError(f"Unknown loading method: {config.loading_method}")

    print(observations.size())
    exit()

    train(observations, gaze_coords, actions)


if __name__ == "__main__":
    main()
