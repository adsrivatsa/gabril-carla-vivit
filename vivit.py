from typing import Tuple

import torch
import torch.nn.functional as Fn
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_flash_attn=True,
        return_last_block_attn: bool = False,
        mask: torch.Tensor = None,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        self.use_flash_attn = use_flash_attn
        self.p_dropout = dropout
        self.scale = dim_head**-0.5
        no_project = heads == 1 and dim_head == dim
        self.return_last_block_attn = return_last_block_attn

        if mask is not None:
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

        self.ln1 = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.split_emb_dim = Rearrange("b t (h d) -> b h t d", h=heads)
        self.drop = nn.Dropout(p=self.p_dropout)
        self.merge_emb_dim = Rearrange("b h t d -> b t (h d)", h=heads)
        self.project = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(p=dropout),
            )
            if not no_project
            else nn.Identity()
        )

    def flash_attn(self, q, k, v, mask=None):
        with sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ]
        ):
            out = Fn.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.p_dropout,
                is_causal=False,
                scale=self.scale,
            )

        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (B, T, E)
        :return: (B, T, E)
        """
        B, T, E = x.shape

        x = self.ln1(x)  # (B, T, E)

        # inner_dim = heads * dim_head, inner dimension before splitting for heads
        q, k, v = self.wq(x), self.wk(x), self.wv(x)  # q, k, v: (B, T, inner_dim)

        q, k, v = map(
            lambda t: self.split_emb_dim(t), [q, k, v]
        )  # q, k, v: (B, heads, T, inner_dim / heads)

        last_block_attn = None
        if self.use_flash_attn and not self.return_last_block_attn:
            out = self.flash_attn(
                q, k, v, mask=self.mask
            )  # (B, heads, T, inner_dim / heads)
        else:
            logits = (
                torch.matmul(q, k.transpose(-1, -2)) * self.scale
            )  # (B, heads, T, T)

            if self.mask is not None:
                if self.mask.dtype == torch.bool:
                    logits = logits.masked_fill(
                        ~self.mask, -float("inf")
                    )  # (B, heads, T, T)
                else:
                    logits = logits + self.mask  # (B, heads, T, T)

            attn = Fn.softmax(logits, dim=-1)  # (B, heads, T, T)

            if self.return_last_block_attn:
                last_block_attn = attn  # (B, heads, T, T)

            attn = self.drop(attn)  # (B, heads, T, T)
            out = torch.matmul(attn, v)  # (B, heads, T, inner_dim / heads)

        out = self.merge_emb_dim(out)  # (B, T, inner_dim)
        out = self.project(out)  # (B, T, E)
        return out, last_block_attn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.l1 = nn.Linear(dim, hidden_dim)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, T, E)
        :return: (B, T, E)
        """
        x = self.ln1(x)  # (B, T, E)
        x = self.l1(x)  # (B, T, hidden_dim)
        x = self.gelu1(x)  # (B, T, hidden_dim)
        x = self.drop1(x)  # (B, T hidden_dim)
        x = self.l2(x)  # (B, T, E)
        x = self.drop2(x)  # (B, T, E)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        return_last_block_attn: bool = False,
        mask: torch.Tensor = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth - 1):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            use_flash_attn=use_flash_attn,
                            return_last_block_attn=False,  # we only want the [CLS] attn from the last block
                            mask=mask,
                        ),
                        FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                    ]
                )
            )
        self.layers.append(
            nn.ModuleList(
                [
                    Attention(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        use_flash_attn=use_flash_attn,
                        return_last_block_attn=return_last_block_attn,
                        mask=mask,
                    ),
                    FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                ]
            )
        )
        self.ln1 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (B, T, E)
        :return: (B, T, E)
        """
        last_block_attn = None
        for attn, ff in self.layers:
            x_attn, last_block_attn = attn(
                x
            )  # x_attn: (B, T, E), last_block_attn: (B, heads, T, T)
            x = x_attn + x  # (B, T, E)
            x = ff(x) + x  # (B, T, E)
        x = self.ln1(x)  # (B, T, E)
        return x, last_block_attn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: Tuple[int, int], channels: int, dim: int):
        """
        patch_size: (PH, PW) of each patch in pixels.
        dim: Embedding dimension E.
        """
        super().__init__()
        ph, pw = patch_size[0], patch_size[1]
        self.patch_dim = ph * pw * channels

        self.patchify = Rearrange(
            "b f c (h ph) (w pw) -> b f (h w) (ph pw c)", ph=ph, pw=pw
        )
        self.ln1 = nn.LayerNorm(self.patch_dim)
        self.l1 = nn.Linear(self.patch_dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, F, C, H, W)
        :return: (B, H / P1, W / P2, E)
        """
        # T = H / PH * W / PW * C, the number of patches/tokens
        x = self.patchify(x)  # (B, F, T, PH * PW * C)
        x = self.ln1(x)  # (B, F, T, PH * PW * C)
        x = self.l1(x)  # (B, F, T, E)
        x = self.ln2(x)  # (B, F, T, E)

        return x  # (B, F, T, E)


class ViViT(nn.Module):
    """
    Video Vision Transformer without frame patching.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        frames: int,
        channels: int,
        n_classes: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        return_cls_attn: bool = False,
    ):
        """
        image_size: (H, W) of the image in pixels.
        patch_size: (PH, PW) of each patch in pixels.
        channels: Image channels C.
        dim: Embedding dimension E.
        """
        super().__init__()

        ih, iw = image_size
        ph, pw = patch_size
        n_patches = ih // ph * iw // pw

        self.patch_emb = PatchEmbedding(patch_size, channels, dim)
        self.pos_enc = nn.Parameter(torch.randn(1, frames, n_patches, dim))
        self.flatten_frames = Rearrange("b f n e -> b (f n) e")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            return_last_block_attn=return_cls_attn,
        )

        self.l1 = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, F, C, H, W)
        :return: (B, num_classes), (B, Heads, F * N)
        """
        B, F, C, H, W = x.shape

        # T = H / PH * W / PW * C, the number of patches/tokens
        x = self.patch_emb(x)  # (B, F, T, E)
        x = x + self.pos_enc  # (B, F, T, E)
        x = self.flatten_frames(x)  # (B, F * T, E)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, F * T + 1, E)
        x, last_block_attn = self.transformer(
            x
        )  # x: (B, F * T + 1, E), last_block_attn: (B, Heads, F * T + 1, F * T + 1)

        # fetching [CLS] token
        x = x[:, 0]  # (B, E)
        x = self.l1(x)  # (B, num_classes)

        cls_attn = last_block_attn[:, :, 0, :]  # (B, Heads, F * T + 1)
        cls_attn = cls_attn[:, :, 1:]  # (B, Heads, F * T)

        return x, cls_attn


class FactorizedViViTV1(nn.Module):
    """
    Factorized Vision Transformer. First transformer does per frame, spatial attention.
    Second transformer does temporal attention on the first transformers [CLS] tokens.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        frames: int,
        channels: int,
        action_space_dim: int,
        dim: int,
        spatial_depth: int,
        temporal_depth: int,
        spatial_heads: int,
        temporal_heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        return_cls_attn: bool = False,
        use_temporal_mask: bool = True,
    ):
        super().__init__()

        ih, iw = image_size
        ph, pw = patch_size
        patches = ih // ph * iw // pw

        self.patch_emb = PatchEmbedding(patch_size, channels, dim)
        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.spatial_pos_enc = nn.Parameter(torch.randn(1, frames, patches + 1, dim))
        self.flatten_frames = Rearrange("b f n e -> (b f) n e")

        self.spatial_transformer = Transformer(
            dim=dim,
            depth=spatial_depth,
            heads=spatial_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            return_last_block_attn=return_cls_attn,
        )

        self.unflatten_attn = Rearrange("(b f) h t1 t2 -> b f h t1 t2", f=frames)
        self.unflatten_frames = Rearrange("(b f) n e -> b f n e", f=frames)
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, dim))
        self.temporal_pos_enc = nn.Parameter(torch.randn(1, frames + 1, dim))

        temporal_mask = None
        if use_temporal_mask:
            # [CLS] should attend to everything
            # other tokens should only attend to past tokens
            temporal_mask = torch.ones((frames + 1, frames + 1), dtype=torch.bool)
            temporal_mask = torch.tril(temporal_mask)
            temporal_mask[0, :] = True

        self.temporal_transformer = Transformer(
            dim=dim,
            depth=temporal_depth,
            heads=temporal_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            return_last_block_attn=False,  # [CLS] vs patch attention only exists in first transformer's attention layers
            mask=temporal_mask,
        )

        self.l1 = nn.Linear(dim, action_space_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (B, F, C, H, W)
        :return: (B, num_classes), (B, F, Heads, T)
        """
        B, F, C, H, W = x.shape

        # T = H / PH * W / PW * C, the number of patches/tokens
        x = self.patch_emb(x)  # (B, F, T, E)
        spatial_cls_tokens = self.spatial_cls_token.expand(B, F, 1, -1)
        x = torch.cat((spatial_cls_tokens, x), dim=2)  # (B, F, T + 1, E)
        x = x + self.spatial_pos_enc  # (B, F, T + 1, E)
        x = self.flatten_frames(x)  # (B * F, T + 1, E)

        x, last_block_attn = self.spatial_transformer(
            x
        )  # x: (B * F, T + 1, E), last_block_attn: (B * F, SpatialHeads, T + 1, T + 1)

        cls_attn = self.unflatten_attn(
            last_block_attn
        )  # (B, F, SpatialHeads, T + 1, T + 1)
        cls_attn = cls_attn[:, :, :, 0, :]  # (B, F, SpatialHeads, T + 1)
        cls_attn = cls_attn[:, :, :, 1:]  # (B, F, SpatialHeads, T)

        x = self.unflatten_frames(x)  # (B, F, T + 1, E)
        x = x[:, :, 0, :]  # (B, F, E)
        temporal_cls_tokens = self.temporal_cls_token.expand(B, 1, -1)
        x = torch.cat((temporal_cls_tokens, x), dim=1)  # (B, F + 1, E)
        x = x + self.temporal_pos_enc  # (B, F + 1, E)

        x, _ = self.temporal_transformer(x)  # (B, F + 1, E)
        x = x[:, 0, :]  # (B, E)
        x = self.l1(x)  # (B, n_classes)

        return x, cls_attn


# class FactorizedViViTV2(nn.Module):
#     """
#     Factorized Vision Transformer. First transformer does per frame, spatial attention and has two [CLS] tokens: [POL] and [GAZE].
#     Second transformer does temporal attention on the first transformers [POL] tokens.
#     """
#
#     def __init__(
#         self,
#         image_size: Tuple[int, int],
#         patch_size: Tuple[int, int],
#         frames: int,
#         channels: int,
#         n_classes: int,
#         dim: int,
#         spatial_depth: int,
#         temporal_depth: int,
#         spatial_heads: int,
#         temporal_heads: int,
#         dim_head: int,
#         mlp_dim: int,
#         dropout: float = 0.0,
#         use_flash_attn: bool = True,
#         return_cls_attn: bool = False,
#         use_temporal_mask: bool = True,
#     ):
#         super().__init__()
#
#         ih, iw = image_size
#         ph, pw = patch_size
#         patches = ih // ph * iw // pw
#
#         self.patch_emb = PatchEmbedding(patch_size, channels, dim)
#         self.spatial_pol_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.spatial_gaze_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.spatial_pos_enc = nn.Parameter(torch.randn(1, frames, patches + 2, dim))
#         self.flatten_frames = Rearrange("b f n e -> (b f) n e")
#
#         self.spatial_transformer = Transformer(
#             dim=dim,
#             depth=spatial_depth,
#             heads=spatial_heads,
#             dim_head=dim_head,
#             mlp_dim=mlp_dim,
#             dropout=dropout,
#             use_flash_attn=use_flash_attn,
#             return_last_block_attn=return_cls_attn,
#         )
#
#         self.unflatten_attn = Rearrange("(b f) h t1 t2 -> b f h t1 t2", f=frames)
#         self.unflatten_frames = Rearrange("(b f) n e -> b f n e", f=frames)
#         self.temporal_cls_token = nn.Parameter(torch.zeros(1, dim))
#         self.temporal_pos_enc = nn.Parameter(torch.randn(1, frames + 1, dim))
#
#         temporal_mask = None
#         if use_temporal_mask:
#             # [POL] and [GAZE] should attend to everything
#             # other tokens should only attend to past tokens
#             temporal_mask = torch.ones((frames + 1, frames + 1), dtype=torch.bool)
#             temporal_mask = torch.tril(temporal_mask)
#             temporal_mask[0, :] = True
#
#         self.temporal_transformer = Transformer(
#             dim=dim,
#             depth=temporal_depth,
#             heads=temporal_heads,
#             dim_head=dim_head,
#             mlp_dim=mlp_dim,
#             dropout=dropout,
#             use_flash_attn=use_flash_attn,
#             return_last_block_attn=False,  # [CLS] vs patch attention only exists in first transformer's attention layers
#             mask=temporal_mask,
#         )
#
#         self.l1 = nn.Linear(dim, n_classes)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         :param x: (B, F, C, H, W)
#         :return: (B, num_classes), (B, F, Heads, T)
#         """
#         B, F, C, H, W = x.shape
#
#         # T = H / PH * W / PW * C, the number of patches/tokens
#         x = self.patch_emb(x)  # (B, F, T, E)
#         pol_tokens = self.spatial_pol_token.expand(B, F, -1, -1)  # (B, F, 1, E)
#         gaze_tokens = self.spatial_gaze_token.expand(B, F, -1, -1)  # (B, F, 1, E)
#         x = torch.cat((pol_tokens, gaze_tokens, x), dim=2)  # (B, F, T + 2, E)
#         x = x + self.spatial_pos_enc  # (B, F, T + 2, E)
#         x = self.flatten_frames(x)  # (B * F, T + 2, E)
#
#         x, last_block_attn = self.spatial_transformer(
#             x
#         )  # x: (B * F, T + 2, E), last_block_attn: (B * F, SpatialHeads, T + 2, T + 2)
#
#         cls_attn = self.unflatten_attn(
#             last_block_attn
#         )  # (B, F, SpatialHeads, T + 2, T + 2)
#         gaze_attn = cls_attn[:, :, :, 1, :]  # (B, F, SpatialHeads, T + 2)
#         gaze_attn = gaze_attn[:, :, :, 2:]  # (B, F, SpatialHeads, T)
#
#         x = self.unflatten_frames(x)  # (B, F, T + 2, E)
#         x = x[:, :, 0, :]  # (B, F, E)
#         temporal_cls_tokens = self.temporal_cls_token.expand(B, 1, -1)
#         x = torch.cat((temporal_cls_tokens, x), dim=1)  # (B, F + 1, E)
#         x = x + self.temporal_pos_enc  # (B, F + 1, E)
#
#         x, _ = self.temporal_transformer(x)  # (B, F + 1, E)
#         x = x[:, 0, :]  # (B, E)
#         x = self.l1(x)  # (B, n_classes)
#
#         return x, gaze_attn
#
#
# class FactorizedViViTV3(nn.Module):
#     """
#     Factorized Vision Transformer. First transformer does per frame, spatial attention and has two [CLS] tokens: [POL] and [GAZE].
#     Second transformer does temporal attention on the first transformers [POL] and [GAZE] tokens.
#     """
#
#     def __init__(
#         self,
#         image_size: Tuple[int, int],
#         patch_size: Tuple[int, int],
#         frames: int,
#         channels: int,
#         n_classes: int,
#         dim: int,
#         spatial_depth: int,
#         temporal_depth: int,
#         spatial_heads: int,
#         temporal_heads: int,
#         dim_head: int,
#         mlp_dim: int,
#         dropout: float = 0.0,
#         use_flash_attn: bool = True,
#         return_cls_attn: bool = False,
#         use_temporal_mask: bool = True,
#     ):
#         super().__init__()
#
#         ih, iw = image_size
#         ph, pw = patch_size
#         patches = ih // ph * iw // pw
#
#         self.patch_emb = PatchEmbedding(patch_size, channels, dim)
#         self.spatial_pol_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.spatial_gaze_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.spatial_pos_enc = nn.Parameter(torch.randn(1, frames, patches + 2, dim))
#         self.flatten_frames = Rearrange("b f n e -> (b f) n e")
#
#         self.spatial_transformer = Transformer(
#             dim=dim,
#             depth=spatial_depth,
#             heads=spatial_heads,
#             dim_head=dim_head,
#             mlp_dim=mlp_dim,
#             dropout=dropout,
#             use_flash_attn=use_flash_attn,
#             return_last_block_attn=return_cls_attn,
#         )
#
#         self.unflatten_attn = Rearrange("(b f) h t1 t2 -> b f h t1 t2", f=frames)
#         self.unflatten_frames = Rearrange("(b f) n e -> b f n e", f=frames)
#         self.temporal_cls_token = nn.Parameter(torch.zeros(1, dim))
#         self.temporal_pos_enc = nn.Parameter(torch.randn(1, frames + 1, dim))
#         self.fusion_proj = nn.Linear(dim * 2, dim)
#         self.fusion_norm = nn.LayerNorm(dim)
#         self.fusion_dropout = nn.Dropout(dropout)
#
#         temporal_mask = None
#         if use_temporal_mask:
#             # [POL] and [GAZE] should attend to everything
#             # other tokens should only attend to past tokens
#             temporal_mask = torch.ones((frames + 1, frames + 1), dtype=torch.bool)
#             temporal_mask = torch.tril(temporal_mask)
#             temporal_mask[0, :] = True
#
#         self.temporal_transformer = Transformer(
#             dim=dim,
#             depth=temporal_depth,
#             heads=temporal_heads,
#             dim_head=dim_head,
#             mlp_dim=mlp_dim,
#             dropout=dropout,
#             use_flash_attn=use_flash_attn,
#             return_last_block_attn=False,  # [CLS] vs patch attention only exists in first transformer's attention layers
#             mask=temporal_mask,
#         )
#
#         self.l1 = nn.Linear(dim, n_classes)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         :param x: (B, F, C, H, W)
#         :return: (B, num_classes), (B, F, Heads, T)
#         """
#         B, F, C, H, W = x.shape
#
#         # T = H / PH * W / PW * C, the number of patches/tokens
#         x = self.patch_emb(x)  # (B, F, T, E)
#         pol_tokens = self.spatial_pol_token.expand(B, F, -1, -1)  # (B, F, 1, E)
#         gaze_tokens = self.spatial_gaze_token.expand(B, F, -1, -1)  # (B, F, 1, E)
#         x = torch.cat((pol_tokens, gaze_tokens, x), dim=2)  # (B, F, T + 2, E)
#         x = x + self.spatial_pos_enc  # (B, F, T + 2, E)
#         x = self.flatten_frames(x)  # (B * F, T + 2, E)
#
#         x, last_block_attn = self.spatial_transformer(
#             x
#         )  # x: (B * F, T + 2, E), last_block_attn: (B * F, SpatialHeads, T + 2, T + 2)
#
#         cls_attn = self.unflatten_attn(
#             last_block_attn
#         )  # (B, F, SpatialHeads, T + 2, T + 2)
#         gaze_attn = cls_attn[:, :, :, 1, :]  # (B, F, SpatialHeads, T + 2)
#         gaze_attn = gaze_attn[:, :, :, 2:]  # (B, F, SpatialHeads, T)
#
#         x = self.unflatten_frames(x)  # (B, F, T + 2, E)
#         pol = x[:, :, 0, :]  # (B, F, E)
#         gaze = x[:, :, 1, :]  # (B, F, E)
#         fused = torch.cat((pol, gaze), dim=-1)  # (B, F, 2 * E)
#         fused = self.fusion_proj(fused)  # (B, F, E)
#         fused = self.fusion_norm(fused)  # (B, F, E)
#         x = self.fusion_dropout(fused)  # (B, F, E)
#
#         temporal_cls_tokens = self.temporal_cls_token.expand(B, 1, -1)
#         x = torch.cat((temporal_cls_tokens, x), dim=1)  # (B, F + 1, E)
#         x = x + self.temporal_pos_enc  # (B, F + 1, E)
#
#         x, _ = self.temporal_transformer(x)  # (B, F + 1, E)
#         x = x[:, 0, :]  # (B, E)
#         x = self.l1(x)  # (B, n_classes)
#
#         return x, gaze_attn
