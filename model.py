import torch.nn as nn
import torch.functional as F


class Encoder(nn.Module):
    def __init__(self, in_shape, embedding_size=256):
        self.conv = nn.Sequential([
            nn.Conv3d(1, 16, 3, 2),
            nn.BatchNorm3d(),
            nn.Conv3d(16, 32, 3, 2),
            nn.BatchNorm3d(),
            nn.Conv3d(32, 32, 3, 2),
            nn.BatchNorm3d(),
            nn.Conv3d(32, 64, 3, 2),
            nn.BatchNorm3d(),
            nn.Conv3d(64, 64, 3, 2),
            nn.BatchNorm3d(),
            nn.Conv3d(64, 128, 3, 2),
            nn.BatchNorm3d(),
            nn.Conv3d(128, 128, 3, 2),
        ])

    def forward(self, img):
        conv_img = self.conv(img)
        batch_size = conv_img.s
        conv_img = conv_img.view((batch_size, -1))
