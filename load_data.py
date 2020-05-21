import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms.functional as tf
import torchvision
from torchvision import transforms
from PIL import Image


class StyleTransferDataSet(data.Dataset):
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


class SRDataSet(data.Dataset):
    def __init__(self, lr_path, hr_path, transform):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.transform = transform

        self.img_names = os.listdir(lr_path)
        for img_name in self.img_names:
            if not (img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')):
                self.img_names.remove(img_name)
        self.img_names = list(set(self.img_names).intersection(set(os.listdir(hr_path))))

    def __getitem__(self, item):
        lr = Image.open(self.lr_path + self.img_names[item]).convert('RGB')
        hr = Image.open(self.hr_path + self.img_names[item]).convert('RGB')
        if random.random() > 0.5:
            lr = tf.hflip(lr)
            hr = tf.hflip(hr)
        if random.random() > 0.5:
            lr = tf.vflip(lr)
            hr = tf.vflip(hr)
        if random.random() > 0.5:
            lr = tf.rotate(lr, 90)
            hr = tf.rotate(hr, 90)
        return self.transform(lr), self.transform(hr)

    def __len__(self):
        return len(self.img_names)


def sr_dataset_test():
    batch_size = 5
    denormalize = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])
    data_set = SRDataSet(lr_path='/Users/soildom/Documents/PycharmProjects/SR/DIV2K/sub_images/train_LR(x4)/',
                         hr_path='/Users/soildom/Documents/PycharmProjects/SR/DIV2K/sub_images/train_HR/',
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])
                         )
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=1)
    for lr, hr in data_loader:
        all_lr = denormalize(lr[0])
        all_hr = denormalize(hr[0])
        for i in range(1, batch_size):
            all_lr = torch.cat((all_lr, denormalize(lr[i])), 1)
            all_hr = torch.cat((all_hr, denormalize(hr[i])), 1)
        break
    torchvision.utils.save_image(all_lr, 'lr_tmp.png')
    torchvision.utils.save_image(all_hr, 'hr_tmp.png')
    print(len(data_set))


if __name__ == '__main__':
    sr_dataset_test()
