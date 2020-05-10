import os
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import cv2 as cv
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms


class COCOTrain2017(data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform

        self.img_names = os.listdir(path)
        for img_name in self.img_names:
            if not (img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')):
                self.img_names.remove(img_name)

    def __getitem__(self, item):
        return self.transform(Image.open(self.path + self.img_names[item]).convert('RGB'))

    def __len__(self):
        return len(self.img_names)


class ResidualBlock(nn.Module):
    def __init__(self, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.Conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128)
        )

    def forward(self, x):
        return x + self.res_scale * self.Conv(x)


class TransferNet(nn.Sequential):
    def __init__(self):
        super(TransferNet, self).__init__(
            nn.Conv2d(3, 32, 9, padding=4, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),

            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, 9, padding=4, padding_mode='reflect')
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        #         nn.init.constant_(m.bias, val=0)


class PerceptualLoss(nn.Module):
    def __init__(self, style, device):
        super(PerceptualLoss, self).__init__()
        self.select = ['0', '5', '10', '19', '21', '28']
        self.vgg = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features)[:29]).to(device).eval()
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


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x):
        return self.l1(x[:, :, 1:, :], x[:, :, :- 1, :]) + self.l1(x[:, :, :, 1:], x[:, :, :, : - 1])


def train(style_img_path):
    batch_size = 4
    epoch_num = 1
    log_size = 200
    content_weight = 1
    style_weight = 1e2
    tv_weight = 1e-6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    style = Image.open(style_img_path).resize((256, 256), Image.BICUBIC)
    style = transform(style).unsqueeze(0).to(device).requires_grad_(False)

    data_set = COCOTrain2017('Data/class_2nd/', transform)
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)

    net = TransferNet().to(device).train()
    perceptual = PerceptualLoss(style, device).to(device).eval()
    tv = TotalVariationLoss().to(device).eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # for epoch in range(1, epoch_num + 1):
    #     for i in range(1, len(data_loader) + 1):
    #         if i % 500 == 0:
    #             print(epoch, i, optimizer.param_groups[0]['lr'])
    #             scheduler.step()

    for epoch in range(1, epoch_num + 1):
        running_content_loss = 0.0
        running_style_loss = 0.0
        running_tv_loss = 0.0
        with tqdm(range(1, len(data_loader) + 1)) as pbar:
            for i, content in zip(pbar, data_loader):
                content = content.to(device).requires_grad_(False)
                x = net(content)
                content_loss, style_loss = perceptual(x, content)
                tv_loss = tv(x)
                loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_content_loss += content_loss.item()
                running_style_loss += style_loss.item()
                running_tv_loss += tv_loss.item()

                pbar.desc = 'Content Loss:%.4f, Style Loss:%f, TV Loss:%.2f ===> ' % (
                    content_loss.item(), style_loss.item(), tv_loss.item())

                if i % log_size == 0:
                    pbar.desc = 'Content Loss:%.4f, Style Loss:%f, TV Loss:%.2f ===> ' % (
                        running_content_loss / log_size, running_style_loss / log_size, running_tv_loss / log_size)
                    print()
                    running_content_loss = 0.0
                    running_style_loss = 0.0
                    running_tv_loss = 0.0

                    with torch.no_grad():
                        net = net.eval()
                        x = Image.open('Data/class_1st/ContentImage/2.jpeg')
                        transform = transforms.Compose([
                            transforms.Resize((x.size[1] // 10, x.size[0] // 10), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        denormalize = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])

                        torchvision.utils.save_image(
                            denormalize(net(transform(x).unsqueeze(0).to(device)).squeeze(0)).clamp_(0, 1), 'tmp.png')
                        net = net.train()

                # if i % 500 == 0:
                #     scheduler.step()

        torch.save(net, 'Model/class_2nd/' + style_img_path.split('/')[-1].split('.')[0] + '.pth')


def generate(src_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = Image.open(src_path)
    transform = transforms.Compose([
        transforms.Resize((x.size[1] // 2, x.size[0] // 2), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    denormalize = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])

    x = transform(x).unsqueeze(0).to(device)
    net = torch.load('Model/class_2nd/Starry_Night.pth', map_location=device)
    y = denormalize(net(x).squeeze(0)).clamp_(0, 1)
    torchvision.utils.save_image(y, 'tmp.png')


if __name__ == '__main__':
    train('Data/class_1st/StyleImage/Starry_Night.jpg')
    # generate('Data/class_1st/ContentImage/1.png')
    # print(nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features)))

    # x = torch.ones((1, 3, 256, 256)).to(torch.device("cuda:0"))
    # net = TransferNet().to(torch.device("cuda:0"))
    # print(net(x).size())
