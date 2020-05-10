import os
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import cv2 as cv
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
    content = cv.imread(content_path)
    style = cv.imread(style_path)
    content = cv.resize(content, (content.shape[1] // 10, content.shape[0] // 10), interpolation=cv.INTER_CUBIC)
    # style = cv.resize(style, (style.shape[1] // 10, style.shape[0] // 10), interpolation=cv.INTER_CUBIC)

    original_h, original_w, _ = content.shape

    cv.imwrite('Output/class_1st/content.jpg', content)
    cv.imwrite('Output/class_1st/style.jpg', style)

    h, w = min(content.shape[0], style.shape[0]), min(content.shape[1], style.shape[1])
    content = cv.resize(content, (w, h), interpolation=cv.INTER_CUBIC).transpose((2, 0, 1))
    style = cv.resize(style, (w, h), interpolation=cv.INTER_CUBIC).transpose((2, 0, 1))

    norm = transforms.Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
    content = torch.from_numpy(content).type(torch.float) / 255
    style = torch.from_numpy(style).type(torch.float) / 255

    return norm(content).unsqueeze(0), norm(style).unsqueeze(0), original_h, original_w,


def generate(content_path, style_path, generate_from_noise=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_step', type=int, default=50000)
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--save_step', type=int, default=100)
    parser.add_argument('--style_weight', type=float, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    config = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists('Output/class_1st/'):
        shutil.rmtree('Output/class_1st/')
    os.makedirs('Output/class_1st/target/')

    content, style, original_h, original_w = load_img(content_path, style_path)
    if generate_from_noise:
        target = torch.randn(content.size())
    else:
        target = content.clone()

    denorm = transforms.Normalize((-1.80, -2.04, -2.12), (4.44, 4.46, 4.37))
    img = target.detach().cpu().squeeze(0)
    img = (denorm(img) * 255).clamp_(0, 255).numpy().transpose((1, 2, 0)).astype(np.uint8)
    img = cv.resize(img, (original_w, original_h), interpolation=cv.INTER_CUBIC)
    cv.imwrite('Output/class_1st/target/t-0.jpg', img)

    content = content.to(device).requires_grad_(False)
    style = style.to(device).requires_grad_(False)
    target = target.to(device).requires_grad_(True)

    vgg = VGGNet().to(device)
    # criterion = nn.MSELoss().to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam([target], lr=config.lr)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 3000, 6000, 10000, 20000, 30000, 40000],
                                                    gamma=0.5)

    with tqdm(range(config.total_step), desc='Class 1st: ') as pbar:
        for step in pbar:
            target_features = vgg(target)
            content_features = vgg(content)
            style_features = vgg(style)

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
                img = target.detach().cpu().squeeze(0)
                img = (denorm(img) * 255).clamp_(0, 255).numpy().transpose((1, 2, 0)).astype(np.uint8)
                img = cv.resize(img, (original_w, original_h), interpolation=cv.INTER_CUBIC)
                cv.imwrite('Output/class_1st/target/t-%d.jpg' % (step + 1), img)


if __name__ == '__main__':
    generate(content_path='ContentImage/2.jpeg', style_path='StyleImage/2.jpeg', generate_from_noise=True)
