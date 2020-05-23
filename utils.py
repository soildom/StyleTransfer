import os
import shutil

import tqdm
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resize(src_path, target_path, size):
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path)

    src_names = os.listdir(src_path)
    for src_name in src_names:
        if not (src_name.endswith('.jpg') or src_name.endswith('.jpeg') or src_name.endswith('.png')):
            src_names.remove(src_name)

    with tqdm.tqdm(range(len(src_names))) as pbar:
        for i, src_name in zip(pbar, src_names):
            pbar.desc = src_name
            src = Image.open(src_path + src_name)
            target = src.resize((size, size), Image.BICUBIC)
            target.save(target_path + src_name)


def show_vgg_features(img_path):
    target_path = 'features/'
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path)

    class VGG19(nn.Module):
        def __init__(self):
            super(VGG19, self).__init__()
            self.select = ['0', '2', '5', '7', '10', '12', '14', '16', '19', '21', '23', '25', '28', '30', '32', '34']
            self.vgg = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features)[:35]).to(device).eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            print(self.vgg)

        def get_features(self, x):
            features = []
            for name, layer in self.vgg._modules.items():
                x = layer(x)
                if name in self.select:
                    features.append(x)
            return self.select, features

    vgg = VGG19()
    x = Image.open(img_path)
    # x = x.resize((x.size[0] // 2, x.size[1] // 2), Image.BICUBIC)
    x = transforms.ToTensor()(x).unsqueeze(0)
    idx, features = vgg.get_features(x)
    for i, feature in zip(idx, features):
        ImageOps.equalize(transforms.ToPILImage()(feature[0, :3, :, :]).convert('L')).save(
            target_path + 'feature' + i + '.png')


if __name__ == '__main__':
    show_vgg_features(img_path='/Users/soildom/Downloads/Image Style Transfer Using Convolutional Neural Networks.jpg')

    # resize('F:/DataSet/COCOtrain2017/', 'Data/COCO-Train2017/', 256)
