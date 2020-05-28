import os
import shutil
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '21', '28']
        self.vgg = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features)[:29]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def load_img(content_path, style_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    content = Image.open(content_path)
    style = Image.open(style_path)

    content = content.resize((content.size[0] // 10, content.size[1] // 10), Image.BICUBIC)
    style = style.resize(content.size, Image.BICUBIC)

    content.save('Output/class_1st/content.jpg')
    style.save('Output/class_1st/style.jpg')

    return transform(content).unsqueeze(0), transform(style).unsqueeze(0)


def generate(content_path, style_path, generate_from_noise=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_step', type=int, default=5000)
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--save_step', type=int, default=50)
    parser.add_argument('--style_weight', type=float, default=50)
    parser.add_argument('--lr', type=float, default=1e-1)
    config = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists('Output/class_1st/'):
        shutil.rmtree('Output/class_1st/')
    os.makedirs('Output/class_1st/target/')

    de_norm = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])

    content, style = load_img(content_path, style_path)

    if generate_from_noise:
        target = torch.randn(content.size())
    else:
        target = content.clone()
    torchvision.utils.save_image(de_norm(target.detach().squeeze(0)), 'Output/class_1st/target/t-0.jpg')

    content = content.to(device).requires_grad_(False)
    style = style.to(device).requires_grad_(False)
    target = target.to(device).requires_grad_(True)

    vgg = VGGNet().to(device)
    # criterion = nn.MSELoss().to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam([target], lr=config.lr)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, int(0.2 * config.total_step), gamma=0.5)

    with torch.no_grad():
        content_features = vgg(content)
        style_features = vgg(style)

    with tqdm(range(config.total_step), desc='Class 1st: ') as pbar:
        for step in pbar:
            target_features = vgg(target)

            style_loss = 0
            content_loss = 0
            for i, (f1, f2, f3) in enumerate(zip(target_features, content_features, style_features)):
                # content loss
                if i == 4:
                    content_loss = criterion(f1, f2)

                # style loss
                b, c, h, w = f1.size()
                f1 = f1.view(b * c, h * w)
                f3 = f3.view(b * c, h * w)
                f1 = torch.mm(f1, f1.t())
                f3 = torch.mm(f3, f3.t())
                style_loss += criterion(f1, f3) / 5

            # content_loss = criterion(target, content)
            pbar.desc = "Content Loss:%.4f, Style Loss:%.4f ====> " % (content_loss.item(), style_loss.item())

            loss = content_loss + config.style_weight * style_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()

            # if (step + 1) % config.log_step == 0:
            #     print('[%5d/%d] ====> Content Loss: %.4f, Style Loss: %.4f' % (
            #         step + 1, config.total_step, content_loss.item(), style_loss.item()))

            if (step + 1) % config.save_step == 0:
                torchvision.utils.save_image(de_norm(target.detach().cpu().squeeze(0)),
                                             'Output/class_1st/target/t-%d.jpg' % (step + 1))


if __name__ == '__main__':
    generate(content_path='ContentImage/IMG_0375.jpeg',
             style_path='StyleImage/神奈川沖浪裏.jpg',
             generate_from_noise=False)
