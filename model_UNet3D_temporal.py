import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        # GroupNorm is more memory/mini-batch friendly than BatchNorm
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3DCore(nn.Module):
    """
    Memory-efficient 3D U-Net core operating on a single 5D volume [B, C, D, H, W].
    Depth is configurable; channel growth is modest to keep memory reasonable.
    """
    def __init__(self, in_channels: int, out_channels: int = 2, base_ch: int = 32, depth: int = 3):
        super().__init__()
        assert depth in (2, 3, 4), "Depth must be 2, 3, or 4 for practicality"
        chs = [base_ch * (2 ** i) for i in range(depth)]  # e.g., [32, 64, 128]

        # Encoder
        self.enc0 = ConvBlock3D(in_channels, chs[0])
        self.down0 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock3D(chs[0], chs[1])
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)

        if depth >= 3:
            self.enc2 = ConvBlock3D(chs[1], chs[2])
            if depth == 4:
                self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)
                self.enc3 = ConvBlock3D(chs[2], chs[3])
        
        # Decoder
        if depth == 2:
            self.up1 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=2, stride=2)
            self.dec1 = ConvBlock3D(chs[0] + chs[0], chs[0])
            dec_in_ch = chs[0]
        elif depth == 3:
            self.up2 = nn.ConvTranspose3d(chs[2], chs[1], kernel_size=2, stride=2)
            self.dec2 = ConvBlock3D(chs[1] + chs[1], chs[1])
            self.up1 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=2, stride=2)
            self.dec1 = ConvBlock3D(chs[0] + chs[0], chs[0])
            dec_in_ch = chs[0]
        else:  # depth == 4
            chs.append(base_ch * (2 ** 3))  # 256
            self.up3 = nn.ConvTranspose3d(chs[3], chs[2], kernel_size=2, stride=2)
            self.dec3 = ConvBlock3D(chs[2] + chs[2], chs[2])
            self.up2 = nn.ConvTranspose3d(chs[2], chs[1], kernel_size=2, stride=2)
            self.dec2 = ConvBlock3D(chs[1] + chs[1], chs[1])
            self.up1 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=2, stride=2)
            self.dec1 = ConvBlock3D(chs[0] + chs[0], chs[0])
            dec_in_ch = chs[0]

        self.head = nn.Conv3d(dec_in_ch, out_channels, kernel_size=1)
        self.depth = depth

    def _align_skip(self, skip: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Center-crop or pad the skip tensor spatially (D,H,W) to match target's spatial size.
        """
        _, _, d_s, h_s, w_s = skip.shape
        _, _, d_t, h_t, w_t = target.shape

        # Pad if smaller
        pd_d = max(0, d_t - d_s)
        pd_h = max(0, h_t - h_s)
        pd_w = max(0, w_t - w_s)
        if pd_d or pd_h or pd_w:
            pad = [
                pd_w // 2, pd_w - pd_w // 2,
                pd_h // 2, pd_h - pd_h // 2,
                pd_d // 2, pd_d - pd_d // 2,
            ]
            skip = F.pad(skip, pad)
            _, _, d_s, h_s, w_s = skip.shape

        # Crop if larger
        cd_d = max(0, d_s - d_t)
        cd_h = max(0, h_s - h_t)
        cd_w = max(0, w_s - w_t)
        if cd_d or cd_h or cd_w:
            d0 = cd_d // 2
            h0 = cd_h // 2
            w0 = cd_w // 2
            skip = skip[:, :, d0:d0 + d_t, h0:h0 + h_t, w0:w0 + w_t]

        return skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)             # B, C0, D, H, W
        x = self.down0(e0)            # B, C0, D/2, H/2, W/2
        e1 = self.enc1(x)             # B, C1, ...

        if self.depth == 2:
            x = e1
        elif self.depth == 3:
            x = self.down1(e1)
            e2 = self.enc2(x)         # B, C2, ...
            # Decoder path
            x = self.up2(e2)
            e1_aligned = self._align_skip(e1, x)
            x = torch.cat([x, e1_aligned], dim=1)
            x = self.dec2(x)
            x = self.up1(x)
            e0_aligned = self._align_skip(e0, x)
            x = torch.cat([x, e0_aligned], dim=1)
            x = self.dec1(x)
            # Align to original input resolution before final conv
            x = self._align_skip(x, e0)
            return self.head(x)
        else:  # depth == 4
            x = self.down1(e1)
            e2 = self.enc2(x)
            x = self.down2(e2)
            e3 = self.enc3(x)
            # Decoder path
            x = self.up3(e3)
            e2_aligned = self._align_skip(e2, x)
            x = torch.cat([x, e2_aligned], dim=1)
            x = self.dec3(x)
            x = self.up2(x)
            e1_aligned = self._align_skip(e1, x)
            x = torch.cat([x, e1_aligned], dim=1)
            x = self.dec2(x)
            x = self.up1(x)
            e0_aligned = self._align_skip(e0, x)
            x = torch.cat([x, e0_aligned], dim=1)
            x = self.dec1(x)
            return self.head(x)

        # depth == 2 decoder
        x = self.up1(e1)
        e0_aligned = self._align_skip(e0, x)
        x = torch.cat([x, e0_aligned], dim=1)
        x = self.dec1(x)
        # Align to original input resolution before final conv
        x = self._align_skip(x, e0)
        return self.head(x)


class TemporalAttention(nn.Module):
    """
    Lightweight temporal attention that computes attention weights over L timesteps
    using global pooled embeddings per timestep. Output is a weighted sum over time.
    """
    def __init__(self, in_channels: int, emb_channels: int = 64, num_heads: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # -> [B, C, 1, 1, 1]
        self.proj_q = nn.Linear(in_channels, emb_channels)
        self.proj_k = nn.Linear(in_channels, emb_channels)
        self.proj_v = nn.Linear(in_channels, emb_channels)
        self.attn = nn.MultiheadAttention(embed_dim=emb_channels, num_heads=num_heads, batch_first=True)
        self.out = nn.Linear(emb_channels, in_channels)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, L, C, D, H, W]
        returns fused: [B, C, D, H, W]
        """
        B, L, C, D, H, W = x_seq.shape
        # Global pooled embeddings per timestep
        pooled = self.pool(x_seq.view(B * L, C, D, H, W)).view(B, L, C)  # [B, L, C]
        q = self.proj_q(pooled)  # [B, L, E]
        k = self.proj_k(pooled)
        v = self.proj_v(pooled)
        # Self-attention over time
        attn_out, _ = self.attn(q, k, v)  # [B, L, E]
        # Use the attended sequence to produce weights per time via projection
        weights = torch.softmax(attn_out.mean(dim=-1), dim=1)  # [B, L]
        # Weighted sum of original high-res volumes along time
        weights = weights.view(B, L, 1, 1, 1, 1)
        fused = (x_seq * weights).sum(dim=1)  # [B, C, D, H, W]
        return fused


class UNet3DTemporal(nn.Module):
    """
    Wrapper that accepts [B, C, D, H, W] or [B, L, C, D, H, W].
    Temporal fusion modes:
      - 'last': use last timestep only (minimal memory)
      - 'mean': average over time
      - 'attn': temporal attention-based weighted sum (lightweight)
    Then runs a compact 3D U-Net on the fused 5D volume.
    """
    def __init__(self,
                 in_channels: int = 12,
                 out_channels: int = 2,
                 base_ch: int = 32,
                 depth: int = 3,
                 temporal_mode: Literal['last', 'mean', 'attn'] = 'attn',
                 attn_heads: int = 4):
        super().__init__()
        self.temporal_mode = temporal_mode
        self.core = UNet3DCore(in_channels=in_channels, out_channels=out_channels, base_ch=base_ch, depth=depth)
        self.temporal_attn: Optional[TemporalAttention] = None
        if temporal_mode == 'attn':
            # Attention operates on input channel dimension before UNet
            self.temporal_attn = TemporalAttention(in_channels=in_channels, emb_channels=min(128, base_ch * 4), num_heads=attn_heads)

    def fuse_time(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C, D, H, W]
        if self.temporal_mode == 'last':
            return x[:, -1]
        elif self.temporal_mode == 'mean':
            return x.mean(dim=1)
        elif self.temporal_mode == 'attn':
            assert self.temporal_attn is not None
            return self.temporal_attn(x)
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, C, D, H, W] or [B, L, C, D, H, W]
        if x.dim() == 5:
            # single timestep
            fused = x
        else:
            assert x.dim() == 6, f"Expected 5D or 6D input, got {x.shape}"
            fused = self.fuse_time(x)
        out = self.core(fused)
        # Ensure exact spatial match with input (handles odd dims/rounding)
        if out.shape[-3:] != fused.shape[-3:]:
            out = F.interpolate(out, size=fused.shape[-3:], mode='trilinear', align_corners=False)
        return out
