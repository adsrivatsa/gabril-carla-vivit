import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass
class Config:
    # paths and flags
    task: str
    carla_dataset_folder: str
    loading_method: str
    use_plots: bool
    save_folder: str
    seed: int
    algorithm: str
    no_gaze: bool

    frame_stack: int
    frame_skip: int

    # gaze
    gaze_sigma: float
    gaze_beta: float
    gaze_alpha: float

    # augmentation
    augment_crop_padding: int
    augment_cutout_hole_size: int
    augment_light_intensity: float
    augment_noise_std: float
    augment_p_pixel_dropout: float
    augment_posterize_bits: int
    augment_blur_pixels: int
    augment_p_spatial_corruptions: float
    augment_p_temporal_corruptions: float

    # transformer
    spatial_patch_size: Tuple[int, int]
    embedding_dim: int
    spatial_depth: int
    temporal_depth: int
    spatial_heads: int
    temporal_heads: int
    inner_dim: int
    mlp_dim: int
    dropout: float
    num_registers: int

    # gaze loss
    gaze_loss_mode: str  # "mean_then_kl" or "kl_then_mean"

    # class imbalance
    use_softmax_weighting: (
        bool  # If True, use inverse-frequency class weights to rectify imbalance
    )

    # hyperparams
    learning_rate: float
    epochs: int
    train_pct: float
    batch_size: int
    lambda_gaze: float
    weight_decay: float
    scheduler_factor: float
    clip_grad_norm: float
    warmup_epochs: int
    warmup_start_factor: float
    min_learning_rate: float

    # validation per epoch
    val_interval: int
    val_episodes: int
    max_episode_length: int

    # testing
    test_episodes: int

    @property
    def run_id(self) -> str:
        """Deterministic hash of training-relevant config fields.
        Used as both the wandb run ID and the checkpoint subfolder name."""
        d = asdict(self)
        # Exclude fields that don't affect training
        for key in ("save_folder", "use_plots"):
            d.pop(key, None)
        # Sort keys for determinism, convert tuples (lists after asdict) back to stable repr
        raw = json.dumps(d, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, default="Mixed_")
parser.add_argument("--carla-dataset-folder", type=str, default="./carla-dataset")
parser.add_argument(
    "--loading-method",
    type=str,
    choices=["mine", "gabril"],
    default="mine",
    help="Data loading method: 'mine' or 'gabril'",
)
parser.add_argument("--use-plots", action="store_true", default=False)
parser.add_argument("--save-folder", type=str, default="./models")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--algorithm",
    type=str,
    choices=["FactorizedViViT", "AuxGazeFactorizedViViT"],
    default="AuxGazeFactorizedViViT",
)
parser.add_argument(
    "--no-gaze",
    action="store_true",
    default=False,
    help="Disable gaze regularization loss. Works with any architecture.",
)
parser.add_argument(
    "--no-softmax-weighting",
    action="store_true",
    default=False,
    help="Disable softmax weighing for class imbalance. When disabled, uses uniform weights.",
)

# frame handling
parser.add_argument("--frame-stack", type=int, default=4)
parser.add_argument("--frame-skip", type=int, default=4)

# gaze parameters
parser.add_argument("--gaze-sigma", type=int, default=15)
parser.add_argument("--gaze-beta", type=float, default=0.99)
parser.add_argument("--gaze-alpha", type=float, default=0.7)

# augmentation parameters
parser.add_argument("--aug-crop-padding", type=int, default=4)
parser.add_argument("--aug-cutout-hole-size", type=int, default=16)
parser.add_argument("--aug-light-intensity", type=float, default=0.2)
parser.add_argument("--aug-noise-std", type=float, default=0.01)
parser.add_argument("--aug-p-pixel-dropout", type=float, default=0.01)
parser.add_argument("--aug-posterize-bits", type=int, default=4)
parser.add_argument("--aug-blur-pixels", type=int, default=2)
parser.add_argument("--aug-p-spatial", type=float, default=0.8)
parser.add_argument("--aug-p-temporal", type=float, default=0.25)

# transformer arch
parser.add_argument(
    "--patch-size", type=int, default=10, help="Spatial patch size (square)"
)
parser.add_argument("--emb-dim", type=int, default=128)
parser.add_argument("--spatial-depth", type=int, default=3)
parser.add_argument("--temporal-depth", type=int, default=1)
parser.add_argument("--spatial-heads", type=int, default=4)
parser.add_argument("--temporal-heads", type=int, default=4)
parser.add_argument("--inner-dim", type=int, default=32)
parser.add_argument("--mlp-dim", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument(
    "--num-registers", type=int, default=0, help="Number of register tokens. Default 0."
)

# hyperparams
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument(
    "--train-pct", type=float, default=0.9999, help="Percentage of data for training"
)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument(
    "--lambda-gaze", type=float, default=0.5, help="Weight for gaze auxiliary loss"
)
parser.add_argument(
    "--gaze-loss-mode",
    type=str,
    choices=["mean_then_kl", "kl_then_mean"],
    default="mean_then_kl",
    help="Gaze loss mode: 'mean_then_kl' averages heads then KL, 'kl_then_mean' computes KL per head then averages",
)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--scheduler-factor", type=float, default=0.5)
parser.add_argument("--clip-grad-norm", type=float, default=1.0)
parser.add_argument("--warmup-epochs", type=int, default=20)
parser.add_argument("--warmup-start-factor", type=float, default=1e-10)
parser.add_argument("--min-lr", type=float, default=1e-6)

# validation
parser.add_argument("--val-interval", type=int, default=100)
parser.add_argument("--val-episodes", type=int, default=100)
parser.add_argument("--max-episode-length", type=int, default=5000)

# testing
parser.add_argument("--test-episodes", type=int, default=100)

args = parser.parse_args()

config = Config(
    # paths and flags
    task=args.task,
    carla_dataset_folder=args.carla_dataset_folder,
    loading_method=args.loading_method,
    use_plots=args.use_plots,
    save_folder=args.save_folder,
    seed=args.seed,
    algorithm=args.algorithm,
    no_gaze=args.no_gaze,
    use_softmax_weighting=not args.no_softmax_weighting,
    frame_stack=args.frame_stack,
    frame_skip=args.frame_skip,
    # gaze
    gaze_sigma=args.gaze_sigma,
    gaze_beta=args.gaze_beta,
    gaze_alpha=args.gaze_alpha,
    # augmentation
    augment_crop_padding=args.aug_crop_padding,
    augment_cutout_hole_size=args.aug_cutout_hole_size,
    augment_light_intensity=args.aug_light_intensity,
    augment_noise_std=args.aug_noise_std,
    augment_p_pixel_dropout=args.aug_p_pixel_dropout,
    augment_posterize_bits=args.aug_posterize_bits,
    augment_blur_pixels=args.aug_blur_pixels,
    augment_p_spatial_corruptions=args.aug_p_spatial,
    augment_p_temporal_corruptions=args.aug_p_temporal,
    # transformer arch
    spatial_patch_size=(args.patch_size, args.patch_size),
    embedding_dim=args.emb_dim,
    spatial_depth=args.spatial_depth,
    temporal_depth=args.temporal_depth,
    spatial_heads=args.spatial_heads,
    temporal_heads=args.temporal_heads,
    inner_dim=args.inner_dim,
    mlp_dim=args.mlp_dim,
    dropout=args.dropout,
    num_registers=args.num_registers,
    # hyperparams
    learning_rate=args.lr,
    epochs=args.epochs,
    train_pct=args.train_pct,
    batch_size=args.batch_size,
    lambda_gaze=args.lambda_gaze,
    gaze_loss_mode=args.gaze_loss_mode,
    weight_decay=args.weight_decay,
    scheduler_factor=args.scheduler_factor,
    clip_grad_norm=args.clip_grad_norm,
    warmup_epochs=args.warmup_epochs,
    warmup_start_factor=args.warmup_start_factor,
    min_learning_rate=args.min_lr,
    # validation per epoch
    val_interval=args.val_interval,
    val_episodes=args.val_episodes,
    max_episode_length=args.max_episode_length,
    # testing
    test_episodes=args.test_episodes,
)
