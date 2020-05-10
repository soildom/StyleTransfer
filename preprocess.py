import os
import shutil

import tqdm
from PIL import Image


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


if __name__ == '__main__':
    resize('E:/Development/DataSet/COCOtrain2017/', 'Data/class_2nd/', 256)
