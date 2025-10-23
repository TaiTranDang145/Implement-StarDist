import torch
import torch.nn as nn
import torch.nn.functional as F

#Conv2d: output_size = (input_size - kernel_size + 2 * padding) / stride + 1
#ConvTranspose2d: output_size = (input_size - 1) * stride + kernel_size - 2 * padding
# 2 conv liên tiếp
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels= in_ch, out_channels= out_ch, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Encoder block (Downsampling)
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch=in_ch, out_ch=out_ch)
        )

    def forward(self, x):
        return self.block(x)


# Decoder block (Upsampling)
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# output convolution
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# stardist architecture
class StarDist(nn.Module):
    def __init__(self, n_channels = 1, n_rays = 32, base_filter = 64):
        super().__init__()
        # Encoder
        # input [B, n_channels, 256, 256]
        self.inc = DoubleConv(n_channels, base_filter) # [B, 64, 256, 256]
        self.down1 = EncoderBlock(base_filter, base_filter*2) # [B, 128, 128, 128]
        self.down2 = EncoderBlock(base_filter*2, base_filter*4) # [B, 256, 64, 64]
        self.down3 = EncoderBlock(base_filter*4, base_filter*8) # [B, 512, 32, 32]
        self.down4 = EncoderBlock(base_filter*8, base_filter*16) # [B, 512, 16, 16]
        # Bottleneck
        #self.bottom = DoubleConv(base_filter*8, base_filter*16) # [B, 1024, 32, 32]

        # Decoder
        self.up1 = DecoderBlock(base_filter*16, base_filter*8) # 512
        self.up2 = DecoderBlock(base_filter*8, base_filter*4) # 256
        self.up3 = DecoderBlock(base_filter*4, base_filter*2) # 128
        self.up4 = DecoderBlock(base_filter*2, base_filter) # 64

        # output heads
        self.prob_head = OutConv(in_ch=base_filter, out_ch=1) # xác suất object
        self.dist_head = OutConv(in_ch=base_filter, out_ch=n_rays) # khoảng cách xuyên tâm

    def forward(self,x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decoder
        x = self.up1(x5,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # heads
        prob = torch.sigmoid(self.prob_head(x))# [B,1,H,W]
        dist = F.relu(self.dist_head(x))        # [B, n_rays,H,W]
        return prob,dist

if __name__ == "__main__":
    model = StarDist(n_channels=3, n_rays=32)
    x = torch.randn(2, 3, 256, 256)
    prob, dist = model(x)
    print("prob:", prob.shape) # B,C,W,H
    print("dist:", dist.shape) # B,n_rays,W,H