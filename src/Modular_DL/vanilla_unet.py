import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding= 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet2D(nn.Module):
        def __init__(self, in_channels=1, out_channels=4):
            super(UNet2D, self).__init__()
            self.inc = DoubleConv2D(in_channels, 64)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(64, 128))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(128, 256))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(256, 512))
            self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(512, 512))

            self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
            self.conv1 = DoubleConv2D(1024, 256)
            self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
            self.conv2 = DoubleConv2D(512, 128)
            self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
            self.conv3 = DoubleConv2D(256, 64)
            self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.conv4 = DoubleConv2D(128, 64)

            self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

        def forward(self, x):
            # x: [B, 1, H, W]
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5)
            x = torch.cat([x4, x], dim=1)
            x = self.conv1(x)

            x = self.up2(x)
            x = torch.cat([x3, x], dim=1)
            x = self.conv2(x)

            x = self.up3(x)
            x = torch.cat([x2, x], dim=1)
            x = self.conv3(x)

            x = self.up4(x)
            x = torch.cat([x1, x], dim=1)
            x = self.conv4(x)

            logits = self.outc(x)
            return logits