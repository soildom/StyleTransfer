from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms
from load_data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, res_scale=0.2):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.Conv = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channel, input_channel, 3, padding=1, padding_mode='reflect'),
        )

    def forward(self, x):
        return x + self.res_scale * self.Conv(x)


class UpSampleBlock(nn.Module):
    def __init__(self, input_channel):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, input_channel, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(F.interpolate(x, scale_factor=2, mode='nearest')))


class SRNet(nn.Sequential):
    def __init__(self):
        super(SRNet, self).__init__(
            nn.Conv2d(3, 64, 9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            UpSampleBlock(64),
            UpSampleBlock(64),

            nn.Conv2d(64, 3, 9, padding=4, padding_mode='reflect')
        )


class SRLoss(nn.Module):
    def __init__(self):
        super(SRLoss, self).__init__()
        self.vgg = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features)[:35]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.l1 = nn.L1Loss()

    def forward(self, hr, sr):
        vgg_loss = self.l1(self.vgg(hr), self.vgg(sr))
        l1_loss = self.l1(hr, sr)
        tv_loss = self.l1(sr[:, :, 1:, :], sr[:, :, :- 1, :]) + self.l1(sr[:, :, :, 1:], sr[:, :, :, : - 1])
        return vgg_loss, l1_loss, tv_loss


def train():
    batch_size = 8
    epoch_num = 100
    log_size = 200
    vgg_weight = 1
    l1_weight = 1e-2
    tv_weight = 1e-8

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_set = SRDataSet(lr_path='D:/PycharmProjects/SR/DIV2K/sub_images/train_LR(x4)/',
                         hr_path='D:/PycharmProjects/SR/DIV2K/sub_images/train_HR/', transform=transform)
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)

    net = SRNet().to(device)
    criterion = SRLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5)

    for epoch in range(1, epoch_num + 1):
        running_vgg_loss = 0.0
        running_l1_loss = 0.0
        running_tv_loss = 0.0

        with tqdm(range(1, len(data_loader) + 1)) as pbar:
            for i, (lr, hr) in zip(pbar, data_loader):
                lr, hr = lr.to(device), hr.to(device)
                sr = net(lr)

                vgg_loss, l1_loss, tv_loss = criterion(hr, sr)
                loss = vgg_weight * vgg_loss + l1_weight * l1_loss + tv_weight * tv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_vgg_loss += vgg_loss.item()
                running_l1_loss += l1_loss.item()
                running_tv_loss += tv_loss.item()

                pbar.desc = 'epoch:%d ===> VGG Loss:%.4f, L1 Loss:%.4f, TV Loss:%.4f ' % (
                    epoch, vgg_loss.item(), l1_loss.item(), tv_loss.item())

                if i % log_size == 0:
                    pbar.desc = 'epoch:%d ===> VGG Loss:%.4f, L1 Loss:%.4f, TV Loss:%.4f ' % (
                        epoch, running_vgg_loss / log_size, running_l1_loss / log_size, running_tv_loss / log_size)
                    print()
                    running_vgg_loss = 0.0
                    running_l1_loss = 0.0
                    running_tv_loss = 0.0

        # print(epoch, optimizer.param_groups[0]['lr'])
        scheduler.step()
        torch.save(net.state_dict(), 'Model/class_2nd/sr.pth')


def generate(src_path):
    x = Image.open(src_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    denormalize = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])

    net = torch.load('Model/class_2nd/sr.pth', map_location=device)
    torchvision.utils.save_image(denormalize(net(transform(x).unsqueeze(0).to(device)).squeeze(0)).clamp_(0, 1),
                                 'tmp.png')


if __name__ == '__main__':
    train()
    # generate('Data/COCO-Train2017/000000000086.jpg')
