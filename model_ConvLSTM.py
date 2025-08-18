import torch
import torch.nn as nn
from typing import Tuple, Optional


class ConvLSTM3DCell(nn.Module):
    """
    A single ConvLSTM cell operating on 3D feature maps (depth, height, width).
    Input/hidden shapes: [B, C, D, H, W]
    """
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv3d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x_t: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # x_t: [B, C_in, D, H, W]
        # state: (h_{t-1}, c_{t-1}) where each is [B, C_hidden, D, H, W]
        B, _, D, H, W = x_t.shape
        if state is None:
            h_prev = torch.zeros(B, self.hidden_channels, D, H, W, device=x_t.device, dtype=x_t.dtype)
            c_prev = torch.zeros(B, self.hidden_channels, D, H, W, device=x_t.device, dtype=x_t.dtype)
        else:
            h_prev, c_prev = state

        gates = self.gates(torch.cat([x_t, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class ConvLSTM3D(nn.Module):
    """
    Multi-layer ConvLSTM operating over a temporal dimension T.
    Expects input of shape [B, T, C, D, H, W].
    Returns the sequence of hidden states from the final layer and the last (h, c).
    """
    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int = 1, kernel_size: int = 3):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_ch = input_channels if l == 0 else hidden_channels
            self.layers.append(ConvLSTM3DCell(in_ch, hidden_channels, kernel_size))

    def forward(self, x: torch.Tensor):
        # x: [B, T, C, D, H, W]
        B, T, C, D, H, W = x.shape
        h: list[torch.Tensor] = []
        c: list[torch.Tensor] = []
        # initialize states as None -> zeros created inside cells on first step
        states: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers

        outputs = []
        for t in range(T):
            x_t = x[:, t]
            for l, cell in enumerate(self.layers):
                h_t, c_t = cell(x_t, states[l])
                states[l] = (h_t, c_t)
                x_t = h_t  # feed to next layer
            outputs.append(x_t)
        # outputs: list of T tensors each [B, hidden, D, H, W]
        out_seq = torch.stack(outputs, dim=1)  # [B, T, hidden, D, H, W]
        h_last, c_last = states[-1]
        return out_seq, (h_last, c_last)


class KappaPredictorConvLSTM(nn.Module):
    """
    ConvLSTM-based predictor for time-series of 3D atmospheric fields.
    - Accepts either [B, C, D, H, W] (single timestep) or [B, T, C, D, H, W].
    - Produces prediction for the last timestep: [B, 2, D, H, W].
    """
    def __init__(self,
                 in_channels: int = 12,
                 hidden_channels: int = 64,
                 num_layers: int = 1,
                 kernel_size: int = 3,
                 out_channels: int = 2):
        super().__init__()
        # Optional light encoder before ConvLSTM
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
        )
        self.convlstm = ConvLSTM3D(input_channels=64,
                                   hidden_channels=hidden_channels,
                                   num_layers=num_layers,
                                   kernel_size=kernel_size)
        # Decoder to 2 targets
        self.dec = nn.Sequential(
            nn.Conv3d(hidden_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: either [B, C, D, H, W] or [B, T, C, D, H, W]
        if x.dim() == 5:
            # Single timestep -> make T=1
            x = x.unsqueeze(1)
        assert x.dim() == 6, f"Expected 6D input [B, T, C, D, H, W], got {x.shape}"

        # Encode each timestep before ConvLSTM to reduce channel size
        B, T, C, D, H, W = x.shape
        x_reshaped = x.view(B * T, C, D, H, W)
        x_enc = self.enc(x_reshaped)
        _, Cenc, Denc, Henc, Wenc = x_enc.shape
        x_enc = x_enc.view(B, T, Cenc, Denc, Henc, Wenc)

        out_seq, (h_last, _) = self.convlstm(x_enc)
        # Use the last hidden state for prediction
        y = self.dec(h_last)
        # y: [B, 2, Denc, Henc, Wenc]
        return y
