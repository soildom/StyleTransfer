import os
import torch.utils.data as data
from PIL import Image


class DataSet(data.Dataset):
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
