import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, use_residual: bool = False, use_se: bool = False):
        super().__init__()
        self.use_residual = use_residual
        self.use_se = use_se
        # Main conv path
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.se = SEBlock3D(out_ch) if use_se else None
        # Residual projection if needed
        self.proj = None
        if use_residual and in_ch != out_ch:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.use_se and self.se is not None:
            out = self.se(out)
        if self.use_residual:
            if self.proj is not None:
                identity = self.proj(identity)
            out = out + identity
        out = self.act2(out)
        return out


class UNet3DCore(nn.Module):
    """
    Memory-efficient 3D U-Net core operating on a single 5D volume [B, C, D, H, W].
    Depth is configurable; channel growth is modest to keep memory reasonable.
    """
    def __init__(self, in_channels: int, out_channels: int = 2, base_ch: int = 32, depth: int = 3,
                 use_residual: bool = False, use_se: bool = False):
        super().__init__()
        assert depth in (2, 3, 4), "Depth must be 2, 3, or 4 for practicality"
        chs = [base_ch * (2 ** i) for i in range(depth)]  # e.g., [32, 64, 128]

        # Encoder
        self.enc0 = ConvBlock3D(in_channels, chs[0], use_residual=use_residual, use_se=use_se)
        self.down0 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock3D(chs[0], chs[1], use_residual=use_residual, use_se=use_se)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)

        if depth >= 3:
            self.enc2 = ConvBlock3D(chs[1], chs[2], use_residual=use_residual, use_se=use_se)
            if depth == 4:
                self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)
                self.enc3 = ConvBlock3D(chs[2], chs[3], use_residual=use_residual, use_se=use_se)
        
        # Decoder
        if depth == 2:
            self.up1 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=2, stride=2)
            self.dec1 = ConvBlock3D(chs[0] + chs[0], chs[0], use_residual=use_residual, use_se=use_se)
            dec_in_ch = chs[0]
        elif depth == 3:
            self.up2 = nn.ConvTranspose3d(chs[2], chs[1], kernel_size=2, stride=2)
            self.dec2 = ConvBlock3D(chs[1] + chs[1], chs[1], use_residual=use_residual, use_se=use_se)
            self.up1 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=2, stride=2)
            self.dec1 = ConvBlock3D(chs[0] + chs[0], chs[0], use_residual=use_residual, use_se=use_se)
            dec_in_ch = chs[0]
        else:  # depth == 4
            chs.append(base_ch * (2 ** 3))  # 256
            self.up3 = nn.ConvTranspose3d(chs[3], chs[2], kernel_size=2, stride=2)
            self.dec3 = ConvBlock3D(chs[2] + chs[2], chs[2], use_residual=use_residual, use_se=use_se)
            self.up2 = nn.ConvTranspose3d(chs[2], chs[1], kernel_size=2, stride=2)
            self.dec2 = ConvBlock3D(chs[1] + chs[1], chs[1], use_residual=use_residual, use_se=use_se)
            self.up1 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=2, stride=2)
            self.dec1 = ConvBlock3D(chs[0] + chs[0], chs[0], use_residual=use_residual, use_se=use_se)
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
                 attn_heads: int = 4,
                 use_residual: bool = False,
                 use_se: bool = False,
                 fuse_multiscale: bool = False):
        super().__init__()
        self.temporal_mode = temporal_mode
        self.core = UNet3DCore(in_channels=in_channels, out_channels=out_channels, base_ch=base_ch, depth=depth,
                               use_residual=use_residual, use_se=use_se)
        self.temporal_attn: Optional[TemporalAttention] = None
        if temporal_mode == 'attn':
            # Attention operates on input channel dimension before UNet
            self.temporal_attn = TemporalAttention(in_channels=in_channels, emb_channels=min(128, base_ch * 4), num_heads=attn_heads)
        self.fuse_multiscale = fuse_multiscale
        self.depth = depth
        self.base_ch = base_ch
        # Projections to match encoder input channels at each deeper scale when using multiscale fusion
        if self.fuse_multiscale:
            ch0 = base_ch
            ch1 = base_ch * 2 if depth >= 3 else None
            ch2 = base_ch * 4 if depth == 4 else None
            # enc1 expects input channels = ch0
            self.ms_proj1 = nn.Conv3d(in_channels, ch0, kernel_size=1, bias=False)
            if depth >= 3:
                # enc2 expects input channels = ch1
                self.ms_proj2 = nn.Conv3d(in_channels, ch1, kernel_size=1, bias=False)
            if depth == 4:
                # enc3 expects input channels = ch2
                self.ms_proj3 = nn.Conv3d(in_channels, ch2, kernel_size=1, bias=False)

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

    def _temporal_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns temporal weights [B, L, 1, 1, 1, 1] according to temporal_mode.
        """
        assert x.dim() == 6, "Expected 6D input for temporal weights"
        B, L = x.shape[0], x.shape[1]
        if self.temporal_mode == 'last':
            w = x.new_zeros((B, L))
            w[:, -1] = 1.0
        elif self.temporal_mode == 'mean':
            w = x.new_full((B, L), 1.0 / L)
        elif self.temporal_mode == 'attn':
            assert self.temporal_attn is not None
            # Reuse attention module to compute weights the same way as in forward
            B_, L_, C, D, H, W = x.shape
            pooled = self.temporal_attn.pool(x.view(B_ * L_, C, D, H, W)).view(B_, L_, C)
            q = self.temporal_attn.proj_q(pooled)
            k = self.temporal_attn.proj_k(pooled)
            v = self.temporal_attn.proj_v(pooled)
            attn_out, _ = self.temporal_attn.attn(q, k, v)
            weights = torch.softmax(attn_out.mean(dim=-1), dim=1)  # [B, L]
            w = weights
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")
        return w.view(B, L, 1, 1, 1, 1)

    def _build_fused_pyramid(self, x_seq: torch.Tensor) -> list:
        """
        Build temporal-fused pyramid per scale using average pooling per timestep
        and attention-derived weights. Returns list of tensors for encoder inputs
        at scales 0..depth-1.
        """
        B, L, C, D, H, W = x_seq.shape
        weights = self._temporal_weights(x_seq)  # [B,L,1,1,1,1]
        fused_list = []
        cur = x_seq
        for s in range(self.depth):
            # Weighted sum over time at current scale
            fused_s = (cur * weights).sum(dim=1)  # [B,C,d,h,w]
            fused_list.append(fused_s)
            if s < self.depth - 1:
                # Downsample each timestep for next scale
                cur = nn.functional.avg_pool3d(cur.view(B * L, C, *cur.shape[-3:]), kernel_size=2, stride=2)
                d2, h2, w2 = cur.shape[-3:]
                cur = cur.view(B, L, C, d2, h2, w2)
        return fused_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, C, D, H, W] or [B, L, C, D, H, W]
        if x.dim() == 5:
            # single timestep
            fused = x
            out = self.core(fused)
            if out.shape[-3:] != fused.shape[-3:]:
                out = F.interpolate(out, size=fused.shape[-3:], mode='trilinear', align_corners=False)
            return out
        else:
            assert x.dim() == 6, f"Expected 5D or 6D input, got {x.shape}"
            if not self.fuse_multiscale:
                fused = self.fuse_time(x)
                out = self.core(fused)
                if out.shape[-3:] != fused.shape[-3:]:
                    out = F.interpolate(out, size=fused.shape[-3:], mode='trilinear', align_corners=False)
                return out
            # Multi-scale temporal fusion path
            pyr = self._build_fused_pyramid(x)  # list len=depth
            core = self.core
            e0 = core.enc0(pyr[0])
            e1_in = pyr[1]
            # Project fused raw inputs at scale-1 to expected channels for enc1
            e1 = core.enc1(self.ms_proj1(e1_in))
            if core.depth == 2:
                xdec = e1
            elif core.depth == 3:
                e2_in = pyr[2]
                # Project to expected channels for enc2
                e2 = core.enc2(self.ms_proj2(e2_in))
                # Decoder
                x = core.up2(e2)
                e1_al = core._align_skip(e1, x)
                x = torch.cat([x, e1_al], dim=1)
                x = core.dec2(x)
                x = core.up1(x)
                e0_al = core._align_skip(e0, x)
                x = torch.cat([x, e0_al], dim=1)
                x = core.dec1(x)
                # Align to original input resolution before final conv
                x = core._align_skip(x, pyr[0])
                out = core.head(x)
                if out.shape[-3:] != pyr[0].shape[-3:]:
                    out = F.interpolate(out, size=pyr[0].shape[-3:], mode='trilinear', align_corners=False)
                return out
            else:  # depth == 4
                e2_in = pyr[2]
                e2 = core.enc2(self.ms_proj2(e2_in))
                e3_in = pyr[3]
                e3 = core.enc3(self.ms_proj3(e3_in))
                # Decoder
                x = core.up3(e3)
                e2_al = core._align_skip(e2, x)
                x = torch.cat([x, e2_al], dim=1)
                x = core.dec3(x)
                x = core.up2(x)
                e1_al = core._align_skip(e1, x)
                x = torch.cat([x, e1_al], dim=1)
                x = core.dec2(x)
                # Final decode to full res
                x = core.up1(x)
                e0_al = core._align_skip(e0, x)
                x = torch.cat([x, e0_al], dim=1)
                x = core.dec1(x)
                x = core._align_skip(x, pyr[0])
                out = core.head(x)
                if out.shape[-3:] != pyr[0].shape[-3:]:
                    out = F.interpolate(out, size=pyr[0].shape[-3:], mode='trilinear', align_corners=False)
                return out
