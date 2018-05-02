import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.testing import randn_like


class Encoder(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.pad = nn.ConstantPad3d((2, 3, 9, 10, 2, 3), 0)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(128),

            nn.Conv3d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(256),

        )

        self.dense = nn.Linear(256, embedding_size)
        self.dense_var = nn.Linear(256, embedding_size)

    def forward(self, img):
        batch_size = img.shape[0]
        img = self.pad(img)
        conv_img = self.conv(img)
        avg_channel = conv_img.view(batch_size, 256, -1).mean(dim=2)
        mean = self.dense(F.dropout(avg_channel, p=0.1))
        log_var = self.dense_var(F.dropout(avg_channel, p=0.1))
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.dense = nn.Linear(embedding_size, 256)

        self.deconv = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ConvTranspose3d(in_channels=256, out_channels=256,
                               kernel_size=3, stride=1, padding=1,
                               output_padding=0, dilation=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=256, out_channels=128,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, dilation=1),
            nn.ReLU(),

            nn.BatchNorm3d(128),
            nn.ConvTranspose3d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1,
                               output_padding=0, dilation=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, dilation=1),
            nn.ReLU(),

            nn.BatchNorm3d(64),
            nn.ConvTranspose3d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1,
                               output_padding=0, dilation=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, dilation=1),
            nn.ReLU(),

            nn.BatchNorm3d(32),
            nn.ConvTranspose3d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1, padding=1,
                               output_padding=0, dilation=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, dilation=1),
            nn.ReLU(),

            nn.BatchNorm3d(16),
            nn.ConvTranspose3d(in_channels=16, out_channels=16,
                               kernel_size=3, stride=1, padding=1,
                               output_padding=0, dilation=1),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=1,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, dilation=1),
        )

    def forward(self, latent):
        batch_size = latent.shape[0]
        avg_channel = self.dense(latent)
        avg_channel = avg_channel[:, :, None,
                      None, None].expand(batch_size, 256, 3, 4, 3) * 1
        rec = self.deconv(avg_channel)

        # self.pad = nn.ConstantPad3d((2, 3, 9, 10, 2, 3), 0)
        rec = rec[:, :, 2:-3, 9:-10, 2:-3]
        return rec


class VAE(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)

    def forward(self, img):
        latent, log_var = self.encoder(img)
        if self.training:
            eps = randn_like(latent)
            latent = latent + torch.exp(log_var / 2) * eps
            penalty = gaussian_kl(latent, log_var)
        else:
            penalty = 0
        return self.decoder(latent), penalty

    def penalty(self):
        return


def gaussian_kl(mean, log_var):
    return torch.mean(- 0.5 * log_var +
                      0.5 * (mean ** 2 + torch.exp(log_var)) - .5)
