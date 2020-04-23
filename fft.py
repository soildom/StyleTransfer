import os
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy.fftpack import fft, ifft


def fft_transfer(content_path, style_path):
    save_path = 'Output/fft/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    content = cv2.imread(content_path)
    style = cv2.imread(style_path)
    style = cv2.resize(style, (content.shape[1], content.shape[0]))
    content = content / 255
    style = style / 255
    y1 = np.array(content)
    y2 = np.array(style)
    fft_y1 = fft(y1)
    fft_y2 = fft(y2)
    abs_y1 = np.abs(fft_y1)  # 取复数的绝对值，即复数的模(双边频谱)
    abs_y2 = np.abs(fft_y2)
    angle_y1 = np.angle(fft_y1)  # 取复数的角度
    angle_y2 = np.angle(fft_y2)
    xfr = np.multiply(abs_y1, np.cos(angle_y2)) + np.multiply(abs_y1, np.sin(angle_y2)) * 1j  # 进行图片的幅度与频度的重新叠加
    yfr = np.multiply(abs_y2, np.cos(angle_y1)) + np.multiply(abs_y2, np.sin(angle_y1)) * 1j
    xr = abs(ifft(xfr))
    yr = abs(ifft(yfr))

    plt.subplot(441)
    plt.imshow(content)
    plt.title('原始图像一', fontsize=9)

    plt.subplot(444)
    plt.imshow(style)
    plt.title('原始图像二', fontsize=9)

    plt.subplot(446)
    plt.imshow(abs_y1)
    plt.title('振幅谱一', fontsize=9)

    plt.subplot(447)
    plt.imshow(abs_y2)
    plt.title('振幅谱二', fontsize=9)

    plt.subplot(449)
    plt.imshow(angle_y1)
    plt.title('相位谱一', fontsize=9)

    plt.subplot(4, 4, 12)
    plt.imshow(angle_y2)
    plt.title('相位谱二', fontsize=9)

    plt.subplot(4, 4, 14)
    plt.imshow(xr)
    plt.title('改变风格后的图像一', fontsize=9)

    plt.subplot(4, 4, 15)
    plt.imshow(yr)
    plt.title('改变风格后的图像二', fontsize=9)

    plt.show()

    cv2.imwrite(save_path + 'Content.png', (content * 255).astype(np.uint8).clip(min=0, max=255))
    cv2.imwrite(save_path + 'Style.png', (style * 255).astype(np.uint8).clip(min=0, max=255))
    cv2.imwrite(save_path + 'Amplitude-1.png', (abs_y1 * 255).astype(np.uint8).clip(min=0, max=255))
    cv2.imwrite(save_path + 'Amplitude-2.png', (abs_y2 * 255).astype(np.uint8).clip(min=0, max=255))
    cv2.imwrite(save_path + 'Phase-1.png', (angle_y1 * 255).astype(np.uint8).clip(min=0, max=255))
    cv2.imwrite(save_path + 'Phase-2.png', (angle_y2 * 255).astype(np.uint8).clip(min=0, max=255))
    cv2.imwrite(save_path + 'Target-1.png', (xr * 255).astype(np.uint8).clip(min=0, max=255))
    cv2.imwrite(save_path + 'Target-2.png', (yr * 255).astype(np.uint8).clip(min=0, max=255))


if __name__ == '__main__':
    fft_transfer('ContentImage/1.png', 'StyleImage/1.jpg')
