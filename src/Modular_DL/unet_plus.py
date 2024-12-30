import torch
import torch.nn as nn
import torch.nn.functional as F

# Nested Convolution Block (for U-Net++)
class NestedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.0):
        super(NestedConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.conv(x)

# U-Net++ Architecture
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, dropout_prob=0.0):
        super(UNetPlusPlus, self).__init__()
        base_ch = 64  # Base number of filters

        # Encoder
        self.enc1 = NestedConvBlock(in_channels, base_ch, dropout_prob)
        self.enc2 = NestedConvBlock(base_ch, base_ch * 2, dropout_prob)
        self.enc3 = NestedConvBlock(base_ch * 2, base_ch * 4, dropout_prob)
        self.enc4 = NestedConvBlock(base_ch * 4, base_ch * 8, dropout_prob)
        self.enc5 = NestedConvBlock(base_ch * 8, base_ch * 8, dropout_prob)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, kernel_size=2, stride=2)
        self.dec1 = NestedConvBlock(base_ch * 16, base_ch * 4, dropout_prob)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)
        self.dec2 = NestedConvBlock(base_ch * 8, base_ch * 2, dropout_prob)

        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.dec3 = NestedConvBlock(base_ch * 4, base_ch, dropout_prob)

        self.up4 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.dec4 = NestedConvBlock(base_ch * 2, base_ch, dropout_prob)

        # Output layer
        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2))
        x5 = self.enc5(F.max_pool2d(x4, 2))

        # Decoder
        d4 = torch.cat([self.up1(x5), x4], dim=1)
        d4 = self.dec1(d4)

        d3 = torch.cat([self.up2(d4), x3], dim=1)
        d3 = self.dec2(d3)

        d2 = torch.cat([self.up3(d3), x2], dim=1)
        d2 = self.dec3(d2)

        d1 = torch.cat([self.up4(d2), x1], dim=1)
        d1 = self.dec4(d1)

        # Output
        return self.outc(d1)
