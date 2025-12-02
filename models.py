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
            nn.Conv2d(in_channels= in_ch, out_channels= out_ch, kernel_size=3,padding=1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, bias = False),
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

    def forward(self, x_dec, x_enc):
        x_dec = self.up(x_dec)
        diffY = x_enc.size(2) - x_dec.size(2)
        diffX = x_enc.size(3) - x_dec.size(3)
        x_dec = F.pad(x_dec, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_enc, x_dec], dim=1)
        return self.conv(x)

# stardist architecture
class StarDist(nn.Module):
    def __init__(self, n_channels = 1, n_rays = 32, base_filters = 32, shared_channels = 128):
        super().__init__()
        self.n_channels = n_channels
        self.n_rays = n_rays
        # Encoder
        # input [B, n_channels, 256, 256]
        self.inc = DoubleConv(n_channels, base_filters) # [B, 32, 256, 256]
        self.down1 = EncoderBlock(base_filters, base_filters*2) # [B, 64, 128, 128]
        self.down2 = EncoderBlock(base_filters*2, base_filters*4) # [B, 128, 64, 64]
        self.down3 = EncoderBlock(base_filters*4, base_filters*8) # [B, 256, 32, 32]
        self.down4 = EncoderBlock(base_filters*8, base_filters*16) # [B, 512, 16, 16]
        # Decoder
        self.up1 = DecoderBlock(base_filters*16, base_filters*8) # 512
        self.up2 = DecoderBlock(base_filters*8, base_filters*4) # 256
        self.up3 = DecoderBlock(base_filters*4, base_filters*2) # 128
        self.up4 = DecoderBlock(base_filters*2, base_filters) # 64

        # add CNN 3 x 3
        self.shared = nn.Sequential(
            nn.Conv2d(base_filters, shared_channels, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True)
        )

        # output heads
        self.prob_head = nn.Conv2d(shared_channels, 1, kernel_size=1) # xác suất object
        self.dist_head = nn.Conv2d(shared_channels, n_rays, kernel_size=1) # khoảng cách xuyên tâm

    def forward(self,x):
        # Encoder
        x1 = self.inc(x)    # [B,  64, H,   W]
        x2 = self.down1(x1) # [B, 128, H/2, W/2]
        x3 = self.down2(x2) # [B, 256, H/4, W/4]
        x4 = self.down3(x3) # [B, 512, H/8, W/8]
        x5 = self.down4(x4) # [B,1024, H/16,W/16]
        # Decoder
        x = self.up1(x5,x4) # [B,512, H/8,  W/8]
        x = self.up2(x, x3) # [B,256, H/4,  W/4]
        x = self.up3(x, x2) # [B,128, H/2,  W/2]
        x = self.up4(x, x1) # [B, 64, H,    W]

        feat = self.shared(x) # [B,128,H,W]

        # heads 
        prob = self.prob_head(feat) # [B,1,H,W]
        dist = self.dist_head(feat) # [B, n_rays,H,W]
        return prob,dist

if __name__ == "__main__":
    model = StarDist(n_channels=3, n_rays=32, base_filters=64, shared_channels=128)
    x = torch.randn(2, 3, 256, 256)
    prob_logits, dist = model(x)
    print("prob_logits:", prob_logits.shape)  # [2, 1, 256, 256]
    print("dist:", dist.shape)         