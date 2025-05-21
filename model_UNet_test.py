import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(3D convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle potential size mismatches due to padding
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=12, n_classes=2, bilinear=False):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Downsampling path (encoder)
        self.inc = DoubleConv3D(in_channels, 32)
        self.down1 = Down3D(32, 64)
        self.down2 = Down3D(64, 128)
        self.down3 = Down3D(128, 256)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down3D(256, 512 // factor)
        
        # Upsampling path (decoder) - adjust output channels to match depth
        self.up1 = Up3D(512, 256 // factor, bilinear)
        self.up2 = Up3D(256, 128 // factor, bilinear)
        self.up3 = Up3D(128, 64 // factor, bilinear)
        self.up4 = Up3D(64, 32, bilinear)
        
        # Output convolution - now outputs 3D predictions
        self.outc = OutConv3D(32, n_classes)
        
        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout3d(0.2)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # Initial double conv
        x2 = self.down1(x1)    # Downsample 1
        x3 = self.down2(x2)    # Downsample 2
        x4 = self.down3(x3)    # Downsample 3
        x5 = self.down4(x4)    # Bottleneck
        
        # Apply dropout in the bottleneck
        x5 = self.dropout(x5)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final 3D output
        logits = self.outc(x)  # [batch_size, n_classes, depth, height, width]
        
        return logits