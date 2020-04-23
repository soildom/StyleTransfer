from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import numpy as np
import cv2
from matplotlib.pylab import mpl


def change_style():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    im1 = cv2.imread('ContentImage/1.png')
    im2 = cv2.imread('StyleImage/1.jpg')
    im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
    im1 = im1 / 255
    im2 = im2 / 255
    y1 = np.array(im1)
    y2 = np.array(im2)
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
    plt.imshow(im1)
    plt.title('原始图像一', fontsize=9)

    plt.subplot(444)
    plt.imshow(im2)
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


change_style()
