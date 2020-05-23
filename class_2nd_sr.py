from tqdm import tqdm
import torch.nn as nn
from load_data import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, res_scale=0.2):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.Conv = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channel, input_channel, 3, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        return x + self.res_scale * self.Conv(x)


class UpSampleBlock(nn.Sequential):
    def __init__(self, input_channel):
        super(UpSampleBlock, self).__init__(
            nn.Conv2d(input_channel, 4 * input_channel, 3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2)
        )


class SRNet(nn.Sequential):
    def __init__(self):
        super(SRNet, self).__init__(
            nn.Conv2d(3, 64, 3, padding=1, padding_mode='reflect'),

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            UpSampleBlock(64),
            UpSampleBlock(64),

            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect'),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class SRLoss(nn.Module):
    def __init__(self):
        super(SRLoss, self).__init__()
        self.vgg = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features)[:35]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.l1 = nn.L1Loss()

    def forward(self, hr, sr):
        vgg_loss = self.l1(self.vgg(hr), self.vgg(sr))
        pixel_loss = self.l1(hr, sr)
        return vgg_loss, pixel_loss


def train():
    batch_size = 8
    epoch_num = 50
    log_size = 200

    vgg_weight = 1
    pixel_weight = 1e-2

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_set = SRDataSet(lr_path='D:/PycharmProjects/SR/DIV2K/sub_images/train_LR(x4)/',
                         hr_path='D:/PycharmProjects/SR/DIV2K/sub_images/train_HR/', transform=transform)
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)

    net = SRNet().to(device).train()

    criterion1 = nn.L1Loss().to(device)
    optimizer1 = torch.optim.Adam(net.parameters(), lr=2e-4)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 5, 0.5)

    criterion2 = SRLoss().to(device)
    optimizer2 = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 5, 0.5)

    for epoch in range(1, epoch_num // 2 + 1):
        running_pixel_loss = 0.0

        with tqdm(range(1, len(data_loader) + 1)) as pbar:
            for i, (lr, hr) in zip(pbar, data_loader):
                lr, hr = lr.to(device), hr.to(device)
                sr = net(lr)

                pixel_loss = criterion1(hr, sr)

                optimizer1.zero_grad()
                pixel_loss.backward()
                optimizer1.step()

                running_pixel_loss += pixel_loss.item()

                pbar.desc = 'stage:1, epoch:%d ===> Pixel Loss:%.4f' % (epoch, pixel_loss.item())

                if i % log_size == 0:
                    pbar.desc = 'stage:1, epoch:%d ===> Pixel Loss:%.4f' % (epoch, running_pixel_loss / log_size)
                    print()
                    running_pixel_loss = 0.0

        scheduler1.step()
        torch.save(net.state_dict(), 'Model/class_2nd/sr.pth')

    for epoch in range(1, epoch_num // 2 + 1):
        running_vgg_loss = 0.0
        running_pixel_loss = 0.0

        with tqdm(range(1, len(data_loader) + 1)) as pbar:
            for i, (lr, hr) in zip(pbar, data_loader):
                lr, hr = lr.to(device), hr.to(device)
                sr = net(lr)

                vgg_loss, pixel_loss = criterion2(hr, sr)
                loss = vgg_weight * vgg_loss + pixel_weight * pixel_loss

                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

                running_vgg_loss += vgg_loss.item()
                running_pixel_loss += pixel_loss.item()

                pbar.desc = 'stage:2, epoch:%d ===> VGG Loss:%.4f, Pixel Loss:%.4f' % (
                    epoch, vgg_loss.item(), pixel_loss.item())

                if i % log_size == 0:
                    pbar.desc = 'stage:2, epoch:%d ===> VGG Loss:%.4f, Pixel Loss:%.4f' % (
                        epoch, running_vgg_loss / log_size, running_pixel_loss / log_size)
                    print()
                    running_vgg_loss = 0.0
                    running_pixel_loss = 0.0

        scheduler2.step()
        torch.save(net.state_dict(), 'Model/class_2nd/sr.pth')


def generate(lr_path):
    with torch.no_grad():
        lr = Image.open(lr_path).convert('RGB')
        lr_bicubic = lr.resize((lr.size[0] * 4, lr.size[1] * 4), Image.BICUBIC)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        denormalize = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])

        net = SRNet()
        net.load_state_dict(torch.load('Model/class_2nd/sr.pth', map_location=device))
        net = net.eval()
        sr = denormalize(net(transform(lr).unsqueeze(0).to(device)).squeeze(0)).clamp_(0, 1)
        torchvision.utils.save_image(torch.cat((transforms.ToTensor()(lr_bicubic), sr), 2), 'tmp.png')


if __name__ == '__main__':
    train()
    # generate('/Users/soildom/Documents/PycharmProjects/SR/DIV2K/sub_images/valid_LR(x4)/0854_8.png')
