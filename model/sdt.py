import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import config_dict
from typing import Optional, Union  # Added for compatibility

# ============================================================
#  Flash-attention import / availability
# ============================================================

try:
    from flash_attn.modules.mha import MHA as FlashMHA

    _FLASH_AVAILABLE = True
except Exception:  # pragma: no cover
    FlashMHA = None
    _FLASH_AVAILABLE = False


class SelfAttention(nn.Module):
    """
    Wrapper that uses flash-attn's MHA when available and requested, otherwise
    falls back to nn.MultiheadAttention.

    Supports:
      - self-attention (cross_attn=False)
      - cross-attention (cross_attn=True)
      - key_padding_mask
      - attn_bias (only in the PyTorch MHA fallback; flash path ignores it)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        *,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.cross_attn = cross_attn

        # Only use flash if explicitly requested and actually available.
        self.use_flash = bool(use_flash_attn and _FLASH_AVAILABLE)

        if self.use_flash:
            # flash-attn MHA; we rely on its own fused projections.
            self.mha_flash = FlashMHA(
                embed_dim=d_model,
                num_heads=nhead,
                cross_attn=cross_attn,
                dropout=dropout,
                use_flash_attn=True,
                # Keep other bells & whistles off for stability:
                qkv_proj_bias=True,
                out_proj_bias=True,
                causal=False,
                fused_bias_fc=False,
            )
            self.mha_torch = None
        else:
            self.mha_flash = None
            self.mha_torch = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        x_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:    [B, N, E] (queries)
        x_kv: [B, N_kv, E] (keys/values) if cross-attn, else None
        """
        # flash-attn path: only if we don't need attn_bias (no API for it).
        if self.use_flash and self.mha_flash is not None and attn_bias is None:
            if self.cross_attn:
                return self.mha_flash(
                    x,
                    x_kv=x_kv if x_kv is not None else x,
                    key_padding_mask=key_padding_mask,
                )
            else:
                return self.mha_flash(
                    x,
                    key_padding_mask=key_padding_mask,
                )

        # Fallback: standard PyTorch MHA (supports attn_bias).
        if self.mha_torch is None:
            # Lazy init if we started with flash-only.
            self.mha_torch = nn.MultiheadAttention(
                self.d_model, self.nhead, dropout=0.0, batch_first=True
            )

        q = x
        if self.cross_attn:
            kv = x_kv if x_kv is not None else x
        else:
            kv = x

        y = self.mha_torch(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False,
        )[0]
        return y


# ============================================================
#  Helpers: time embedding, blocks, local mixers, hybrid heads
# ============================================================

def _build_sinusoidal_table(E: int, device):
    inv = torch.exp(-math.log(10_000) * torch.arange(0, E, 2, device=device) / E)
    table = torch.zeros(1, E, device=device)
    table[:, 0::2] = inv
    table[:, 1::2] = inv
    return table


class _SinTimeSigma(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("_freq", _build_sinusoidal_table(dim, "cpu"))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: [B]
        freq = self._freq.to(sigma.device)
        phases = sigma.log()[:, None] * freq
        emb = torch.empty_like(phases)
        emb[:, 0::2] = phases[:, 0::2].cos()
        emb[:, 1::2] = phases[:, 1::2].sin()
        return emb  # [B, E]


class SwiGLU(nn.Module):
    """
    x -> (A x) ⊙ SiLU(B x) -> W_o
    Set hidden ≈ 2/3 * dim_ff to keep FLOPs close to classic GELU-4E FFN.
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.Wa = nn.Linear(dim, hidden, bias=True)
        self.Wb = nn.Linear(dim, hidden, bias=True)
        self.Wo = nn.Linear(hidden, dim, bias=True)
        self.drop = nn.Dropout(dropout)
        # init
        for m in (self.Wa, self.Wb, self.Wo):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.Wa(x)
        b = F.silu(self.Wb(x))
        y = a * b
        y = self.drop(y)
        return self.Wo(y)


class AdaLNZero(nn.Module):
    """
    Produces (scale, shift, gate) from a time embedding and applies to a normalized stream.
    Zero-initialized so each block starts as identity (like U-ViT).
    """

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d, 3 * d))
        # zero-init last linear so (scale,shift,gate)=0 at start
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor):
        """
        h: [B, N, E]
        t_emb: [B, E]
        Returns: h_mod [B,N,E], gate [B,E]
        """
        s, b, g = self.mlp(t_emb).chunk(3, dim=-1)  # [B,E] each
        h = F.layer_norm(h, h.shape[-1:])  # per-token norm
        return (1 + s).unsqueeze(1) * h + b.unsqueeze(1), g  # broadcast across N


class PreNormBlockAda(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_swiglu: bool = False,
        *,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.dim_ff = dim_ff or (4 * d_model)

        self.attn = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            cross_attn=False,
            use_flash_attn=use_flash_attn,
        )

        if use_swiglu:
            hidden = int(2 * self.dim_ff / 3)
            self.ff = SwiGLU(d_model, hidden, dropout=dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, self.dim_ff),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.dim_ff, d_model),
            )

        self.drop = nn.Dropout(dropout)
        self.adaln1 = AdaLNZero(d_model)
        self.adaln2 = AdaLNZero(d_model)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        h, gate = self.adaln1(x, t_emb)
        y = self.attn(
            h,
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
        )
        x = x + self.drop(y) * gate.unsqueeze(1)

        h, gate = self.adaln2(x, t_emb)
        y = self.ff(h)
        x = x + self.drop(y) * gate.unsqueeze(1)
        return x


class PreNormBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: Optional[int] = None,
        dropout: float = 0.0,
        *,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.dim_ff = dim_ff or (4 * d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            cross_attn=False,
            use_flash_attn=use_flash_attn,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        h = self.norm1(x)
        y = self.attn(
            h,
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
        )
        x = x + self.drop(y)
        y = self.ff(self.norm2(x))
        return x + self.drop(y)


class RelPosBias1D(nn.Module):
    """
    Shared-head 1D relative positional bias for patch tokens.
    For a window W, we learn an embedding over relative offsets in [-W+1, W-1].
    Bias is added to attention logits (before softmax) for every head.
    """

    def __init__(self, max_distance: int = 64):
        super().__init__()
        assert max_distance >= 1
        self.max_distance = int(max_distance)
        self.num_buckets = 2 * self.max_distance - 1  # offsets [-W+1, ..., 0, ..., W-1]
        self.emb = nn.Embedding(self.num_buckets, 1)  # shared across heads

        nn.init.normal_(self.emb.weight, mean=0.0, std=1e-4)

    def forward(self, N: int, device=None, dtype=None) -> torch.Tensor:
        if N <= 1:
            return torch.zeros((N, N), device=device, dtype=dtype)

        idx = torch.arange(N, device=device)
        rel = idx[:, None] - idx[None, :]  # [N,N] in [-N+1, N-1]
        rel = rel.clamp(-self.max_distance + 1, self.max_distance - 1)
        rel_bucket = rel + (self.max_distance - 1)  # map to [0, 2W-2]
        bias = self.emb(rel_bucket)  # [N,N,1]
        return bias.squeeze(-1).to(dtype)  # [N,N]


class LocalSequenceMixer(nn.Module):
    """
    Local mixer in sequence space (depthwise + pointwise 1D conv).
    """

    def __init__(
        self, d_model: int, kernel_size: int = 9, dropout: float = 0.0, mixer_type: str = "conv"
    ):
        super().__init__()
        self.mixer_type = mixer_type
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        if mixer_type == "conv":
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    padding=pad,
                    groups=d_model,
                ),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
        else:
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mixer_type == "conv":
            y = x.transpose(1, 2)  # [B,E,S]
            y = self.net(y)        # [B,E,S]
            y = y.transpose(1, 2)  # [B,S,E]
            return x + self.dropout(y)
        else:
            return x


class HybridSequenceHeadV2(nn.Module):
    """
    New configurable hybrid head for sequence denoising.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        out_dim: int,
        pos_dim: int,
        content_dim: int,
        noisy_dim: int = 1,
        kernel_size: int = 9,
        dropout: float = 0.0,
        use_cross_attn: bool = True,
        use_local_mixer: bool = False,
        use_self_attn: bool = True,
        *,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_cross_attn = use_cross_attn
        self.use_local_mixer = use_local_mixer
        self.use_self_attn = use_self_attn

        self.content_dim = content_dim
        self.noisy_dim = noisy_dim
        in_channels = content_dim + noisy_dim  # x_denoised + x_noisy

        # Previously: Linear(2 + pos_dim, d_model) with scalar x_denoised/x_noisy.
        self.seq_proj = nn.Linear(in_channels + pos_dim, d_model)

        if use_cross_attn:
            self.adaln_cross = AdaLNZero(d_model)
            self.cross_attn = SelfAttention(
                d_model,
                nhead,
                dropout=dropout,
                cross_attn=True,
                use_flash_attn=use_flash_attn,
            )
        else:
            self.adaln_cross = None
            self.cross_attn = None

        if use_self_attn:
            self.adaln_self = AdaLNZero(d_model)
            self.self_attn = SelfAttention(
                d_model,
                nhead,
                dropout=dropout,
                cross_attn=False,
                use_flash_attn=use_flash_attn,
            )
        else:
            self.adaln_self = None
            self.self_attn = None

        if use_local_mixer:
            self.local_mixer = LocalSequenceMixer(
                d_model=d_model,
                kernel_size=kernel_size,
                dropout=dropout,
                mixer_type="conv",
            )
        else:
            self.local_mixer = None

        self.adaln_ff = AdaLNZero(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x_denoised: torch.Tensor,          # [B,S,C_d]
        x_noisy: torch.Tensor,             # [B,S,C_n]
        pos_feats: torch.Tensor,           # [1,S,pos_dim]
        patch_tokens: Optional[torch.Tensor], # [B,N,E] or None
        t_emb: torch.Tensor,               # [B,E]
        pad_mask_tokens: Optional[torch.Tensor] = None,  # [B,N] or None
    ) -> torch.Tensor:
        B, S, _ = x_denoised.shape

        # Concatenate along feature dimension
        content = torch.cat([x_denoised, x_noisy], dim=-1)   # [B,S,C_d + C_n]

        x_aug = torch.cat(
            [content, pos_feats.expand(B, -1, -1)],
            dim=-1,
        )  # [B,S, C_d + C_n + pos_dim]
        seq = self.seq_proj(x_aug)  # [B,S,E]

        if self.cross_attn is not None and patch_tokens is not None:
            h, _ = self.adaln_cross(seq, t_emb)
            y = self.cross_attn(
                h,
                x_kv=patch_tokens,
                key_padding_mask=pad_mask_tokens,
            )
            seq = seq + self.drop(y)

        if self.self_attn is not None:
            h, _ = self.adaln_self(seq, t_emb)
            y = self.self_attn(h)
            seq = seq + self.drop(y)

        if self.local_mixer is not None:
            seq = self.local_mixer(seq)

        h, gate_ff = self.adaln_ff(seq, t_emb)
        y = self.ff(h)
        seq = seq + self.drop(y) * gate_ff.unsqueeze(1)

        logits = self.out(seq)  # [B,S,out_dim]
        return logits


class OptimalSkipMLPHead(nn.Module):
    """
    Final Optimized Head for Semantic Bits.
    """

    def __init__(
        self,
        d_model: int,       # Trunk dimension (E)
        patch_size: int,    # Patch Size (P)
        out_dim: int,       # Output dimension (usually 1)
        content_dim: int,   # Dimension of x_denoised (C)
        noisy_dim: int,     # Dimension of x_noisy (C)
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.P = patch_size
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # 1. Patch Adapter: E -> P * H
        # Creates P unique vectors from one patch token (bit-aware global context).
        self.patch_adapter = nn.Linear(d_model, patch_size * hidden_dim)

        # 2. Input Adapter: 2*C -> H
        # Compresses [x_recon, x_noisy] into hidden_dim space.
        in_channels = content_dim + noisy_dim
        self.input_adapter = nn.Linear(in_channels, hidden_dim)

        # 3. Time Projection: E -> H
        # Projects global time embedding (d_model) down to hidden_dim for AdaLN.
        self.time_proj = nn.Linear(d_model, hidden_dim)

        # 4. AdaLNZero in hidden_dim space
        self.adaln = AdaLNZero(hidden_dim)

        # 5. Output MLP: hidden_dim -> hidden_dim -> out_dim
        self.mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # Initialization
        nn.init.xavier_uniform_(self.patch_adapter.weight)
        nn.init.zeros_(self.patch_adapter.bias)

        nn.init.xavier_uniform_(self.input_adapter.weight)
        nn.init.zeros_(self.input_adapter.bias)

        nn.init.normal_(self.time_proj.weight, 0.0, 0.02)
        nn.init.zeros_(self.time_proj.bias)

        for m in self.mixer:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_denoised: torch.Tensor,   # [B,S,C]
        x_noisy: torch.Tensor,      # [B,S,C]
        patch_tokens: torch.Tensor, # [B,N,E]
        t_emb: torch.Tensor,        # [B,E]
    ) -> torch.Tensor:

        B, S, _ = x_noisy.shape
        B2, N, _ = patch_tokens.shape
        assert B == B2, "Batch size mismatch between x_noisy and patch_tokens"

        # --- A. Global Context (from trunk patch tokens) ---
        # [B, N, E] -> [B, N, P*H]
        global_flat = self.patch_adapter(patch_tokens)

        # [B, N, P*H] -> [B, N*P, H] = [B, S_pad, H]
        global_feat = global_flat.view(B, N * self.P, self.hidden_dim)

        # Align to actual sequence length S (handle padding/truncation)
        if global_feat.shape[1] > S:
            global_feat = global_feat[:, :S, :]
        elif global_feat.shape[1] < S:
            global_feat = F.pad(global_feat, (0, 0, 0, S - global_feat.shape[1]))

        # --- B. Local Context (skip connection) ---
        # [x_denoised, x_noisy] concat along channel dim: [B,S,2C]
        local_input = torch.cat([x_denoised, x_noisy], dim=-1)
        local_feat = self.input_adapter(local_input)   # [B,S,H]

        # --- C. Fusion (ResNet-style) ---
        # Combine global + local features
        h = global_feat + local_feat                   # [B,S,H]

        # --- D. Time Conditioning with gating ---
        # Project time embedding [B,E] -> [B,H]
        t_emb_proj = self.time_proj(t_emb)             # [B,H]

        # AdaLNZero: returns modulated features + gate
        # h_norm: [B,S,H], gate: [B,H]
        h_norm, gate = self.adaln(h, t_emb_proj)

        # Use gate to modulate the hidden features.
        # Broadcast gate over sequence dimension: [B,1,H] -> [B,S,H]
        h_gated = h_norm * gate.unsqueeze(1)          # [B,S,H]

        # --- E. Predict per bit ---
        logits = self.mixer(h_gated)                  # [B,S,out_dim]

        return logits


# ============================================================
#   MAIN MODEL: SequenceVDTContinuousModelv2
# ============================================================

class SequenceVDTContinuousModelv2(nn.Module):
    """
    VDT-style diffusion denoiser generalized to 1D sequences (sequence-only).
    """

    def __init__(self, cfg: config_dict.ConfigDict):
        super().__init__()
        self.cfg = cfg

        assert cfg.framework in ("continuous_score", "discrete_sedd")
        self.is_discrete = (cfg.framework == "discrete_sedd")

        # ---- read config ----
        P = int(cfg.model.patch_size)
        E = int(cfg.model.hidden_dim)
        L = int(cfg.model.n_blocks)
        H = int(cfg.model.n_heads)
        OD = int(getattr(cfg.model, "out_dim", 1))
        dropout = float(getattr(cfg.model, "dropout", 0.0))
        head_type = str(getattr(cfg.model, "head_type", "hybrid_attn_v2")).lower()
        use_adaln = bool(getattr(cfg.model, "use_adaln", True))
        use_swiglu = bool(getattr(cfg.model, "use_swiglu", False))
        use_flash_attn = bool(getattr(cfg.model, "use_flash_attn", False))

        assert head_type in {
            "patch_bits_sedd",
            "hybrid_attn_v2",
            "optimal_skip_mlp",
        }, f"Unsupported head_type: {head_type}"

        self.P, self.E, self.out_dim = P, E, OD
        self.head_type = head_type

        # --- content channel dimension per branch ---
        cd_disc = int(getattr(cfg.model, "content_dim_discrete", 1))
        cd_cont = int(getattr(cfg.model, "content_dim_continuous", 1))

        if self.is_discrete:
            self.C = cd_disc
            self.token_embed = nn.Embedding(cfg.data.vocab_size, self.C)
            self.q_matrix_type = cfg.diffusion.discrete.q_matrix_type.lower()
            self.scale_by_sigma = bool(getattr(cfg.model, "scale_by_sigma", True))
            self.cont_input_proj = None
        else:
            self.C = cd_cont
            self.token_embed = None
            self.q_matrix_type = "none"
            self.scale_by_sigma = False

            if self.C == 1:
                self.cont_input_proj = None
            else:
                self.cont_input_proj = nn.Linear(1, self.C)

        # ----- Fourier positional features -----
        self.n_fourier_global = int(getattr(cfg.model, "n_fourier_global", 8))
        self.n_fourier_local = int(getattr(cfg.model, "n_fourier_local", 4))
        self.pos_dim = (2 * self.n_fourier_global) + (2 * self.n_fourier_local) + 1

        self.C_total = self.C + self.pos_dim
        self.patch_dim = self.C_total * self.P

        # ----- Relative Positional Bias over patch tokens -----
        md = int(getattr(cfg.model, "rpb_max_distance", 64))
        self.rpb = None if md <= 1 else RelPosBias1D(md)

        # If we have RPB (attn_bias != None), disable flash in encoder blocks
        # because flash-attn MHA doesn't support an additive attn mask.
        if use_flash_attn and self.rpb is not None:
            print(
                "[FlashAttn] rpb_max_distance > 1 → encoder blocks need attn_bias; "
                "disabling flash-attn in encoder blocks."
            )
            use_flash_attn_blocks = False
        else:
            use_flash_attn_blocks = use_flash_attn

        # σ embedding
        self.time_fn = _SinTimeSigma(self.E)
        self.time_proj = nn.Linear(self.E, self.E, bias=False)
        self.time_cond = nn.Linear(self.E, self.E)

        # patch ↔ token projections
        self.patch_proj = nn.Linear(self.patch_dim, self.E)
        self.unpatch_proj = nn.Linear(self.E, self.patch_dim)

        # transformer encoder
        dim_ff = int(getattr(cfg.model, "dim_ff", 4 * E))
        if use_adaln:
            self.blocks = nn.ModuleList(
                [
                    PreNormBlockAda(
                        self.E,
                        H,
                        dim_ff=dim_ff,
                        dropout=dropout,
                        use_swiglu=use_swiglu,
                        use_flash_attn=use_flash_attn_blocks,
                    )
                    for _ in range(L)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    PreNormBlock(
                        self.E,
                        H,
                        dim_ff=dim_ff,
                        dropout=dropout,
                        use_flash_attn=use_flash_attn_blocks,
                    )
                    for _ in range(L)
                ]
            )

        # ----- HEADS -----
        if self.head_type == "patch_bits_sedd":
            # E_head can equal E, or be smaller (configurable)
            self.E_head = int(getattr(cfg.model, "head_hidden_dim", self.E))

            # From patch token [E] → P * E_head per patch
            self.patch_to_bits = nn.Linear(self.E, self.P * self.E_head)

            self.bit_norm = nn.LayerNorm(self.E_head)
            self.bit_out = nn.Linear(self.E_head, self.out_dim)

            nn.init.normal_(self.patch_to_bits.weight, 0.0, 0.02)
            nn.init.zeros_(self.patch_to_bits.bias)
            nn.init.normal_(self.bit_out.weight, 0.0, 0.02)
            nn.init.zeros_(self.bit_out.bias)

        elif self.head_type == "hybrid_attn_v2":
            local_kernel = int(getattr(cfg.model, "head_kernel", 3))
            if local_kernel % 2 == 0:
                local_kernel += 1

            use_local_mixer = bool(
                getattr(cfg.model, "head_use_local_mixer", False)
            )
            use_cross_attn = bool(
                getattr(cfg.model, "head_use_cross_attn", True)
            )
            use_self_attn = bool(
                getattr(cfg.model, "head_use_self_attn", True)
            )

            # self.C is the content channel dim:
            #   - discrete: content_dim_discrete (e.g. 384-d token embeddings)
            #   - continuous: content_dim_continuous (e.g. 16-d bit embeddings)
            #
            # For head_v2 we feed:
            #   x_denoised: x_recon        [B,S,C]
            #   x_noisy:    x_skip_content [B,S,C]
            #
            # => both branches have dimension C.
            self.head = HybridSequenceHeadV2(
                d_model=self.E,
                nhead=H,
                out_dim=self.out_dim,
                pos_dim=self.pos_dim,
                content_dim=self.C,
                noisy_dim=self.C,
                kernel_size=local_kernel,
                dropout=dropout,
                use_cross_attn=use_cross_attn,
                use_local_mixer=use_local_mixer,
                use_self_attn=use_self_attn,
                use_flash_attn=use_flash_attn,
            )

        elif self.head_type == "optimal_skip_mlp":
            hidden_dim = int(getattr(cfg.model, "head_hidden", 256))

            self.head = OptimalSkipMLPHead(
                d_model=self.E,
                patch_size=self.P,
                out_dim=self.out_dim,
                content_dim=self.C,
                noisy_dim=self.C,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        # ----- initialization -----
        for m in (self.patch_proj, self.unpatch_proj, self.time_cond):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.time_proj.weight, 0.0, 0.02)

        if isinstance(getattr(self, "head", None), nn.Linear):
            nn.init.normal_(self.head.weight, 0.0, 0.02)
            nn.init.zeros_(self.head.bias)

        if self.cont_input_proj is not None:
            nn.init.normal_(self.cont_input_proj.weight, 0.0, 0.02)
            nn.init.zeros_(self.cont_input_proj.bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_content(
        self, x_t: torch.Tensor, sigma: torch.Tensor
    ):
        """
        x_t:   [B,S]
        sigma: [B]

        Returns:
            content:         [B,S,C]
            x_skip_content:  [B,S,C]
            x_noisy_for_head:[B,S]
        """
        if self.is_discrete:
            x_tokens = x_t.long()                        # [B,S]
            content = self.token_embed(x_tokens)         # [B,S,C_disc]
            x_skip_content = content
            x_noisy_for_head = x_tokens.float()
            return content, x_skip_content, x_noisy_for_head

        # Continuous branch (EDM-style scaling)
        sigma_data = self.cfg.diffusion.continuous.sigma_data
        c_in = 1.0 / (sigma.pow(2) + sigma_data**2).sqrt()   # [B]
        x_t_scaled = x_t.float() * c_in.view(-1, 1)          # [B,S]

        if self.cont_input_proj is None:
            content = x_t_scaled.unsqueeze(-1)               # [B,S,1]
        else:
            content = self.cont_input_proj(
                x_t_scaled.unsqueeze(-1)
            )                                                # [B,S,C_cont]

        x_skip_content   = content
        x_noisy_for_head = x_t_scaled
        return content, x_skip_content, x_noisy_for_head

    def _fourier_pos_feats(self, S: int, device, dtype):
        """
        Returns positional features of shape [1, S, pos_dim]
        (global sin/cos, local-in-patch sin/cos, and distance-to-patch-center).
        """
        t32 = torch.float32
        pi2 = 2.0 * math.pi

        # global Fourier on absolute index s
        s = torch.arange(S, device=device, dtype=t32)  # [S]
        denom_g = max(S - 1, 1)
        if self.n_fourier_global > 0:
            k_g = torch.arange(
                1, self.n_fourier_global + 1, device=device, dtype=t32
            )
            omega_g = pi2 * k_g / denom_g
            ss = s[:, None] * omega_g[None, :]  # [S,G]
            global_feats = torch.cat([ss.sin(), ss.cos()], dim=-1)  # [S,2G]
        else:
            global_feats = torch.zeros(S, 0, device=device, dtype=t32)

        # local Fourier on in-patch index ℓ = s mod P
        if self.n_fourier_local > 0:
            l = (torch.arange(S, device=device, dtype=t32) % self.P)  # [S]
            denom_l = max(self.P - 1, 1)
            k_l = torch.arange(
                1, self.n_fourier_local + 1, device=device, dtype=t32
            )
            omega_l = pi2 * k_l / denom_l
            ll = l[:, None] * omega_l[None, :]  # [S,L]
            local_feats = torch.cat([ll.sin(), ll.cos()], dim=-1)  # [S,2L]
        else:
            local_feats = torch.zeros(S, 0, device=device, dtype=t32)

        center = (self.P - 1) / 2.0
        dist_center = (
            ((torch.arange(S, device=device, dtype=t32) % self.P) - center).abs()
            / max(center, 1e-6)
        ).unsqueeze(-1)  # [S,1]

        pos = torch.cat([global_feats, local_feats, dist_center], dim=-1)  # [S,pos_dim]
        return pos.unsqueeze(0).to(dtype)  # [1,S,pos_dim]

    def _append_pos(self, content: torch.Tensor) -> torch.Tensor:
        """
        [B,S,C] → [B,S,C_total] = concat(content, fourier-pos).
        """
        B, S, _ = content.shape
        pos = self._fourier_pos_feats(S, content.device, content.dtype)  # [1,S,pos_dim]
        return torch.cat([content, pos.expand(B, -1, -1)], dim=-1)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        """x: [B,S,D] → pad S to multiple of P (zeros)."""
        B, S, D = x.shape
        r = S % multiple
        pad_len = 0 if r == 0 else (multiple - r)
        if pad_len == 0:
            return x, 0
        pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1), pad_len

    def _patchify(self, x: torch.Tensor):
        """[B,S_pad,D] → [B,N,patch_dim], with N=S_pad/P."""
        B, S, D = x.shape
        assert S % self.P == 0
        N = S // self.P
        return x.view(B, N, self.P * D), N

    def _unpatchify(self, patches: torch.Tensor, S_pad: int):
        """
        [B,N,patch_dim] → [B,S_pad,C_total] then drop positional channels
        to return content channels only: [B,S_pad,C].
        """
        B, N, PD = patches.shape
        assert PD == self.patch_dim and N * self.P == S_pad
        x = patches.view(B, S_pad, self.C_total)
        return x[:, :, : self.C]

    def _postprocess_logits(
        self,
        logits: torch.Tensor,  # [B,S,out_dim]
        x_t: torch.Tensor,
        sigma: torch.Tensor,
    ):
        """Map trunk logits to the final output for each framework."""
        if not self.is_discrete:
            return logits

        x = logits  # [B,S,V]

        if self.scale_by_sigma and self.q_matrix_type == "absorb":
            # Prevent potential division by zero/log(0) issues in compiled graph
            esigm1 = torch.where(
                sigma < 0.5,
                torch.expm1(sigma),
                sigma.exp() - 1,
            )
            # Ensure safe broadcasting and dtype
            esigm1_log = esigm1.log().to(x.dtype)[:, None, None]
            
            # Using shape[2] instead of shape[-1] helps dynamo shape inference
            V = x.shape[2]
            x = x - esigm1_log - math.log(V - 1)

        # ----------------------------------------------------------------------
        # CRITICAL FIX FOR TORCH.COMPILE SEGFAULT
        # ----------------------------------------------------------------------
        # 1. Ensure indices are strictly LongTensor
        indices = x_t.long().unsqueeze(-1)
        
        # 2. Use in-place scatter_ instead of functional torch.scatter.
        #    The functional version with a scalar src=0.0 often generates 
        #    invalid Triton kernels when combined with AMP (FP16).
        # 3. Use a float literal compatible with the tensor's dtype implicitly.
        x.scatter_(-1, indices, 0.0)
        
        return x

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x_t:   [B, S]
        sigma:[B]
        """
        assert x_t.dim() == 2 and sigma.dim() == 1 and x_t.size(0) == sigma.size(0)
        B, S_orig = x_t.shape

        # 0) unified: build content + skips + head noisy view
        content, x_skip_content, x_noisy_for_head = self._build_content(x_t, sigma)
        x_aug = self._append_pos(content)  # [B,S_orig,C_total]

        # 1) pad to multiple of P
        x_pad, pad_len = self._pad_to_multiple(x_aug, self.P)  # [B,S_pad,C_total]
        S_pad = x_pad.size(1)

        # 2) patch-level key_padding_mask
        pad_mask_tokens = None
        if pad_len > 0:
            pad_mask_seq = torch.zeros(
                B, S_pad, dtype=torch.bool, device=x_pad.device
            )
            pad_mask_seq[:, S_pad - pad_len :] = True
            N = S_pad // self.P
            pad_mask_tokens = pad_mask_seq.view(B, N, self.P).all(dim=-1)  # [B,N]
        else:
            N = S_pad // self.P

        # 3) patchify + project
        tokens_in, N = self._patchify(x_pad)   # [B,N,patch_dim]
        tokens = self.patch_proj(tokens_in)    # [B,N,E]

        # 3a) time embedding
        t_emb = self.time_cond(self.time_proj(self.time_fn(sigma)))  # [B,E]

        # 3b) relative positional bias over PATCH TOKENS
        attn_bias = (
            None
            if self.rpb is None
            else self.rpb(N, device=tokens.device, dtype=tokens.dtype)
        )

        # 4) transformer blocks
        for blk in self.blocks:
            if isinstance(blk, PreNormBlockAda):
                tokens = blk(
                    tokens,
                    t_emb,
                    key_padding_mask=pad_mask_tokens,
                    attn_bias=attn_bias,
                )
            else:
                tokens = blk(
                    tokens,
                    key_padding_mask=pad_mask_tokens,
                    attn_bias=attn_bias,
                )

        # 5) heads → logits [B,S,out_dim]
        if self.head_type == "patch_bits_sedd":
            # tokens: [B,N,E]
            B, N, E = tokens.shape
            assert E == self.E

            bits_flat = self.patch_to_bits(tokens)                # [B,N,P*E_head]
            bits = bits_flat.view(B, N * self.P, self.E_head)     # [B,S_pad,E_head]

            if pad_len > 0:
                bits = bits[:, :S_orig, :]                        # [B,S_orig,E_head]

            logits = self.bit_out(self.bit_norm(bits))            # [B,S,out_dim]

        elif self.head_type == "hybrid_attn_v2":
            # New head: takes full C-dim denoised and noisy embeddings.
            #
            #   x_denoised = x_recon        [B,S,C] (from trunk)
            #   x_noisy    = x_skip_content [B,S,C] (embedding of noisy input)
            #
            patches_out = self.unpatch_proj(tokens)               # [B,N,patch_dim]
            x_recon_pad = self._unpatchify(patches_out, S_pad)    # [B,S_pad,C]
            x_recon = (
                x_recon_pad[:, :S_orig, :]
                if pad_len > 0
                else x_recon_pad
            ).contiguous()                                        # [B,S,C]

            # Use the skip content as "noisy embedding" for the head:
            #  - discrete: token_embed(x_t_noisy)        [B,S,C_disc]
            #  - continuous: embedding of scaled x_t     [B,S,C_cont]
            x_noisy = x_skip_content.to(x_recon.dtype)           # [B,S,C]

            pos = self._fourier_pos_feats(S_orig, x_recon.device, x_recon.dtype)

            logits = self.head(
                x_denoised=x_recon,           # [B,S,C]
                x_noisy=x_noisy,              # [B,S,C]
                pos_feats=pos,
                patch_tokens=tokens,
                t_emb=t_emb,
                pad_mask_tokens=pad_mask_tokens,
            )
        
        elif self.head_type == "optimal_skip_mlp":
            # 1. Unpatch to get trunk reconstruction (x_denoised)
            patches_out = self.unpatch_proj(tokens)           # [B,N,patch_dim]
            x_recon_pad = self._unpatchify(patches_out, S_pad) # [B,S_pad,C]

            x_recon = (
                x_recon_pad[:, :S_orig, :]                    # [B,S_orig,C]
                if pad_len > 0
                else x_recon_pad
            ).contiguous()

            # 2. Noisy input embedding from skip connection
            # x_skip_content: [B,S_pad,C] from _build_content
            x_noisy = x_skip_content[:, :S_orig, :].to(x_recon.dtype)  # [B,S,C]

            # 3. Call the optimal skip MLP head
            logits = self.head(
                x_denoised=x_recon,        # [B,S,C]
                x_noisy=x_noisy,           # [B,S,C]
                patch_tokens=tokens,       # [B,N,E]
                t_emb=t_emb,               # [B,E]
            )

        # 6) final framework-specific post-processing
        return self._postprocess_logits(logits, x_t, sigma)