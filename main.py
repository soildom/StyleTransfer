import os
import imageio
import cv2 as cv
from fft import fft_transfer
from vgg import vgg_transfer


def img2gif(img_root_path, save_path, duration=0.0001):
    """
    :param duration: 图像间隔时间
    """
    img_names = os.listdir(img_root_path)
    img_index = list()
    for img_name in img_names:
        if not (img_name.endswith('.png') or img_name.endswith('jpg') or img_name.endswith('.jpeg')):
            img_names.remove(img_name)
        else:
            img_index.append(int(img_name.split('.')[0].split('-')[1]))
    img_index.sort()

    frames = list()
    for idx in img_index:
        print(idx)
        img = cv.imread(img_root_path + ('t-%d.jpg' % idx))
        cv.imwrite('tmp.jpg', cv.resize(img, (img.shape[1], img.shape[0])))
        frames.append(imageio.imread('tmp.jpg'))
    os.remove('tmp.jpg')
    imageio.mimsave(save_path, frames, 'GIF', duration=duration)


if __name__ == '__main__':
    content_path = 'ContentImage/1.png'
    style_path = 'StyleImage/1.jpg'

    # fft_transfer(content_path, style_path)
    # vgg_transfer(content_path, style_path)

    img2gif('Output/vgg/target/', 'test.gif')
