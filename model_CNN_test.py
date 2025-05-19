import torch.nn as nn

class KappaPredictorCNN(nn.Module):
    def __init__(self, in_channels):
        super(KappaPredictorCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 2, kernel_size=1)  # Output: kappa_SU, kappa_CA
        )

    def forward(self, x):  # x: [B, C, L, H, W]
        return self.decoder(self.encoder(x))