import os
import shutil

import cv2
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt


def multi_channel_fft(img):
    img_fft = np.zeros(img.shape, dtype=np.complex128)
    for i in range(img.shape[2]):
        img_fft[:, :, i] = fft2(img[:, :, i])
    return img_fft


def multi_channel_ifft(img_fft):
    img = np.zeros(img_fft.shape, dtype=np.complex128)
    for i in range(img_fft.shape[2]):
        img[:, :, i] = ifft2(img_fft[:, :, i])
    return img


def fft_transfer(content_path, style_path):
    dpi = 400
    fontsize = 5
    save_path = 'Output/fft/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    content = cv2.imread(content_path) / 255.0
    style = cv2.resize(cv2.imread(style_path), (content.shape[1], content.shape[0])) / 255.0

    content_fft = multi_channel_fft(content)
    style_fft = multi_channel_fft(style)

    content_amplitude = np.abs(content_fft)  # 取复数的绝对值，即复数的模(双边频谱)
    style_amplitude = np.abs(style_fft)

    content_phase = np.angle(content_fft)  # 取复数的角度
    style_phase = np.angle(style_fft)

    # 进行图片的幅度与频度的重新叠加
    target1_fft = content_amplitude * (np.cos(style_phase) + 1j * np.sin(style_phase))
    target2_fft = style_amplitude * (np.cos(content_phase) + 1j * np.sin(content_phase))
    target1 = np.abs(multi_channel_ifft(target1_fft)).clip(min=0, max=1)
    target2 = np.abs(multi_channel_ifft(target2_fft)).clip(min=0, max=1)

    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.dpi'] = dpi

    plt.subplot(4, 4, 5)
    plt.imshow(content[:, :, ::-1])
    plt.title('Content', fontsize=fontsize, pad=0)
    plt.axis('off')

    plt.subplot(4, 4, 8)
    plt.imshow(style[:, :, ::-1])
    plt.title('Style', fontsize=fontsize, pad=0)
    plt.axis('off')

    plt.subplot(4, 4, 2)
    plt.imshow(np.log(content_amplitude[:, :, 0]), 'gray')
    plt.title('Content Amplitude B', fontsize=fontsize, pad=0)
    plt.axis('off')
    plt.subplot(4, 4, 6)
    plt.imshow(np.log(content_amplitude[:, :, 1]), 'gray')
    plt.title('Content Amplitude G', fontsize=fontsize, pad=0)
    plt.axis('off')
    plt.subplot(4, 4, 10)
    plt.imshow(np.log(content_amplitude[:, :, 2]), 'gray')
    plt.title('Content Amplitude R', fontsize=fontsize, pad=0)
    plt.axis('off')

    plt.subplot(4, 4, 3)
    plt.imshow(np.log(style_amplitude[:, :, 0]), 'gray')
    plt.title('Style Amplitude B', fontsize=fontsize, pad=0)
    plt.axis('off')
    plt.subplot(4, 4, 7)
    plt.imshow(np.log(style_amplitude[:, :, 1]), 'gray')
    plt.title('Style Amplitude G', fontsize=fontsize, pad=0)
    plt.axis('off')
    plt.subplot(4, 4, 11)
    plt.imshow(np.log(style_amplitude[:, :, 2]), 'gray')
    plt.title('Style Amplitude R', fontsize=fontsize, pad=0)
    plt.axis('off')

    plt.subplot(4, 4, 14)
    plt.imshow(target1[:, :, ::-1])
    plt.title('Target-1', fontsize=fontsize, pad=0)
    plt.axis('off')

    plt.subplot(4, 4, 15)
    plt.imshow(target2[:, :, ::-1])
    plt.title('Target-2', fontsize=fontsize, pad=0)
    plt.axis('off')

    plt.savefig(save_path + 'Plot.png', bbox_inches='tight')
    plt.show()

    cv2.imwrite(save_path + 'Content.png', (content * 255).astype(np.uint8))
    cv2.imwrite(save_path + 'Style.png', (style * 255).astype(np.uint8))
    cv2.imwrite(save_path + 'Target-1.png', (target1 * 255).astype(np.uint8))
    cv2.imwrite(save_path + 'Target-2.png', (target2 * 255).astype(np.uint8))


if __name__ == '__main__':
    fft_transfer('ContentImage/IMG_0375.jpeg', 'StyleImage/神奈川沖浪裏.jpg')
