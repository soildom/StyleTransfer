import os
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Conv2d_MetaNet(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(Conv2d_MetaNet, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             bias, padding_mode)

        for param in self.parameters():
            param.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.Conv = nn.Sequential(
            Conv2d_MetaNet(32, 32, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            Conv2d_MetaNet(32, 32, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(32)
        )

    def forward(self, x):
        return x + self.Conv(x)


class TransferNet(nn.Sequential):
    def __init__(self):
        super(TransferNet, self).__init__(
            nn.Conv2d(3, 8, 9, padding=4, padding_mode='reflect'),
            nn.InstanceNorm2d(8),
            nn.ReLU(inplace=True),

            Conv2d_MetaNet(8, 16, 3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),

            Conv2d_MetaNet(16, 32, 3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2d_MetaNet(32, 16, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv2d_MetaNet(16, 8, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 3, 9, padding=4, padding_mode='reflect')
        )

    def set_weight(self, weight):
        pass


class MetaNet(nn.Module):
    def __init__(self, style, device):
        super(MetaNet, self).__init__()
        self.select = ['3', '8', '15', '22']
        self.vgg = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features)[:23]).to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        self.style_features = self.get_features(style)

    def forward(self, x, content):
        x_features = self.get_features(x)
        content_features = self.get_features(content)

        style_loss = 0.0
        for i, (f1, f2, f3) in enumerate(zip(x_features, content_features, self.style_features)):
            # content loss
            if i == 4:
                content_loss = self.criterion(f1, f2)

            # style loss
            b, c, h, w = f1.size()
            f1 = f1.view(b, c, h * w)
            f3 = f3.view(1, c, h * w)
            f1_gram = torch.bmm(f1, f1.transpose(1, 2))  # / (c * h * w)
            f3_gram = torch.bmm(f3, f3.transpose(1, 2))  # / (c * h * w)
            style_loss += self.criterion(f1_gram, f3_gram.expand_as(f1_gram)) / 6

        return content_loss, style_loss

    def get_features(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


if __name__ == '__main__':
    # print(torchvision.models.vgg19(pretrained=True).features)
    x = torch.zeros((1, 3, 256, 256))
    net = TransferNet()
    print(net(x).size())
