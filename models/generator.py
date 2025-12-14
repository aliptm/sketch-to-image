"""
Custom Generator Model - U-Net Architecture
Transforms sketches into detailed images
"""
import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    """U-Net building block with optional downsampling/upsampling"""
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        self.down = down
        
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        if not self.down:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    """
    Custom U-Net Generator for sketch-to-image translation
    Architecture:
    - Encoder: Downsampling path with skip connections
    - Decoder: Upsampling path with skip connections
    """
    def __init__(self, input_channels=1, output_channels=3, features=64):
        super(Generator, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = UNetBlock(features, features * 2, down=True)
        self.enc3 = UNetBlock(features * 2, features * 4, down=True)
        self.enc4 = UNetBlock(features * 4, features * 8, down=True)
        self.enc5 = UNetBlock(features * 8, features * 8, down=True)
        self.enc6 = UNetBlock(features * 8, features * 8, down=True)
        self.enc7 = UNetBlock(features * 8, features * 8, down=True)
        
        # Bottleneck (process features at deepest level without changing size)
        # After 7 downsamplings: 256->128->64->32->16->8->4->2, so e7 is 2x2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 8, features * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling with skip connections)
        # Channel progression: bottleneck(512) -> d1(512) + e6(512) = 1024 -> d2(512) + e5(512) = 1024 -> 
        # d3(512) + e4(512) = 1024 -> d4(512) + e3(256) = 768 -> d5(256) + e2(128) = 384 -> d6(128) + e1(64) = 192 -> d7(64)
        self.dec1 = UNetBlock(features * 8, features * 8, down=False, use_dropout=True)  # 512 -> 512
        self.dec2 = UNetBlock(features * 8 * 2, features * 8, down=False, use_dropout=True)  # 1024 -> 512
        self.dec3 = UNetBlock(features * 8 * 2, features * 8, down=False, use_dropout=True)  # 1024 -> 512
        self.dec4 = UNetBlock(features * 8 * 2, features * 8, down=False)  # 1024 -> 512
        self.dec5 = UNetBlock(features * 8 + features * 4, features * 4, down=False)  # 768 -> 256 (512+256 from e3)
        self.dec6 = UNetBlock(features * 4 + features * 2, features * 2, down=False)  # 384 -> 128 (256+128 from e2)
        self.dec7 = UNetBlock(features * 2 + features, features, down=False)  # 192 -> 64 (128+64 from e1)
        # Final layer: d7 is already 256x256 with features channels, so just convert to output channels
        self.dec8 = nn.Sequential(
            nn.Conv2d(features, output_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        # Bottleneck (process at 2x2 level)
        bottleneck = self.bottleneck(e7)
        
        # Decoder with skip connections
        # Size progression: e7(2x2) -> d1(4x4) -> d2(8x8) -> d3(16x16) -> d4(32x32) -> d5(64x64) -> d6(128x128) -> d7(256x256)
        # Skip connections: d1(4x4)<->e6(4x4), d2(8x8)<->e5(8x8), d3(16x16)<->e4(16x16), d4(32x32)<->e3(32x32), d5(64x64)<->e2(64x64), d6(128x128)<->e1(128x128)
        d1 = self.dec1(bottleneck)  # 2x2 -> 4x4
        d1 = torch.cat([d1, e6], dim=1)  # e6 is 4x4
        
        d2 = self.dec2(d1)  # 4x4 -> 8x8
        d2 = torch.cat([d2, e5], dim=1)  # e5 is 8x8
        
        d3 = self.dec3(d2)  # 8x8 -> 16x16
        d3 = torch.cat([d3, e4], dim=1)  # e4 is 16x16
        
        d4 = self.dec4(d3)  # 16x16 -> 32x32
        d4 = torch.cat([d4, e3], dim=1)  # e3 is 32x32
        
        d5 = self.dec5(d4)  # 32x32 -> 64x64
        d5 = torch.cat([d5, e2], dim=1)  # e2 is 64x64
        
        d6 = self.dec6(d5)  # 64x64 -> 128x128
        d6 = torch.cat([d6, e1], dim=1)  # e1 is 128x128
        
        d7 = self.dec7(d6)  # 128x128 -> 256x256
        # d7 is already 256x256, so dec8 should not upsample further
        # Instead, use a final conv to get output channels
        output = self.dec8(d7)
        
        return output


def test_generator():
    """Test function to verify generator architecture"""
    model = Generator(input_channels=1, output_channels=3, features=64)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Generator parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_generator()

