import os
import shutil
import argparse
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import utils


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
    original_h, original_w, _ = content.shape

    cv.imwrite('Output/content.png', content)
    cv.imwrite('Output/style.png', style)

    h, w = min(content.shape[0], style.shape[0]), min(content.shape[1], style.shape[1])
    content = cv.resize(content, (w, h), interpolation=cv.INTER_CUBIC).transpose((2, 0, 1))
    style = cv.resize(style, (w, h), interpolation=cv.INTER_CUBIC).transpose((2, 0, 1))

    norm = transforms.Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
    content = torch.from_numpy(content).type(torch.float) / 255
    style = torch.from_numpy(style).type(torch.float) / 255

    return norm(content).unsqueeze(0), norm(style).unsqueeze(0), original_h, original_w,


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='ContentImage/1.png')
    parser.add_argument('--style', type=str, default='StyleImage/1.jpg')
    parser.add_argument('--total_step', type=int, default=5000)
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--style_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    config = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists('Output/'):
        shutil.rmtree('Output/')
    os.makedirs('Output/target/')

    content, style, h, w = load_img(config.content, config.style)
    target = content.clone()

    content = content.requires_grad_(False).to(device)
    style = style.requires_grad_(False).to(device)
    target = target.requires_grad_(True).to(device)

    vgg = VGGNet().to(device)
    mse_sum = nn.MSELoss(reduction='sum').to(device)
    mse_mean = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam([target], lr=config.lr, betas=(0.5, 0.999))

    for step in range(config.total_step):
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)

        style_loss = 0
        content_loss = 0
        for i, (f1, f2, f3) in enumerate(zip(target_features, content_features, style_features)):
            # 计算content损失
            if i == 4:
                content_loss = mse_sum(f1, f2) / 2

            # 计算style损失
            b, c, h, w = f1.size()
            f1 = f1.view(b * c, h * w)
            f3 = f3.view(b * c, h * w)
            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())
            style_loss += mse_mean(f1, f3) / 5

        loss = content_loss + config.style_weight * style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % config.log_step == 0:
            print('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
                  .format(step + 1, config.total_step, content_loss.item(), style_loss.item()))

            denorm = transforms.Normalize((-1.80, -2.04, -2.12), (4.44, 4.46, 4.37))
            img = target.clone().squeeze(0)
            img = (denorm(img) * 255).clamp_(0, 255).numpy().transpose((1, 2, 0)).astype(np.uint8)
            img = cv.resize(img, (w, h), interpolation=cv.INTER_CUBIC)
            cv.imwrite('Output/target/t-{}.png'.format(step + 1), img)
