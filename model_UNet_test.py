import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)        # [B, C, H, W]
        x2 = self.pool(x1)
        x2 = self.enc2(x2)
        return x2  # Bottleneck features per time step

class TemporalUNet(nn.Module):
    def __init__(self, in_ch=12, out_ch=2, hidden_dim=128):
        super().__init__()
        self.encoder = UNetEncoder(in_ch)
        
        # Assuming bottleneck gives [B, C, H/4, W/4]
        self.temporal_lstm = nn.LSTM(input_size=hidden_dim,
                                     hidden_size=hidden_dim,
                                     batch_first=True)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        feats = []

        for t in range(T):
            enc = self.encoder(x[:, t])  # [B, hidden_dim, H', W']
            feats.append(enc.flatten(2)) # Flatten spatial dims

        feats = torch.stack(feats, dim=1)  # [B, T, hidden_dim * H' * W']
        feats = feats.mean(-1)  # [B, T, hidden_dim] as simplification

        lstm_out, _ = self.temporal_lstm(feats)  # [B, T, hidden_dim]
        last = lstm_out[:, -1]  # Use last timestep

        # Reshape back to spatial (H', W')
        B, H_, W_ = B, H//4, W//4
        spatial_feat = last.view(B, -1, H_, W_)  # [B, hidden_dim, H', W']

        out = self.decoder(spatial_feat)
        return out  # [B, out_ch, H, W]
