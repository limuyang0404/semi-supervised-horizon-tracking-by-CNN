# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from collections import Counter
import scipy
from scipy import signal
import segyio
import pandas as pd
import matplotlib.pyplot as plt
def Gaussian_conv(img, kernel):
    output = 0
    for i in range(3):
        for j in range(3):
            output += img[i, j]*kernel[i, j]
    return output
def Gaussain_filter(img, kernel):
    img_padding = np.zeros(shape=(img.shape[0]+2, img.shape[1]+2), dtype='float32')
    img_padding[1:-1, 1:-1] = img
    output = np.zeros(shape=img.shape, dtype='float32')
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = Gaussian_conv(img_padding[i:i+3, j:j+3], kernel)

    return output
def Gaussain_kernel_generate(deviation):
    output = []
    for i in range(3):
        for j in range(3):
            a = 1 / (2 * np.pi * deviation ** 2) * np.exp((-1) * ((1-i) ** 2 + (1-j) ** 2) / (2*deviation**2))
            output.append(a)
    output = np.array(output).reshape(3, 3)
    return output
def spectrum(img, kernel):
    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)
    # fimg = np.log(np.abs(fshift))
    fimg = Gaussain_filter(img, kernel)
    plt.subplot(121)
    plt.imshow(np.moveaxis(img, 0, -1), 'gray')
    plt.title('img')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(np.moveaxis(fimg, 0, -1), 'gray')
    plt.title('fourier')
    plt.axis('off')
    plt.show()
    return
def input_data_norm(input_cube_img):
    input_cube_img_min = np.min(input_cube_img)
    input_cube_img_max = np.max(input_cube_img)
    input_cube_img_range = input_cube_img_max - input_cube_img_min
    input_cube_img_norm = (input_cube_img - input_cube_img_min) / input_cube_img_range
    return input_cube_img_norm
def hilbert_transform(data_cube):
    """
    Instantaneous phase
    :param data_cube:seismic amplitude
    :return:Instantaneous phase
    """
    output = np.zeros(shape=data_cube.shape, dtype='float32')
    for i in range(data_cube.shape[0]):
        for j in range(data_cube.shape[1]):
            ft = data_cube[i, j, :]
            gt = np.imag(scipy.signal.hilbert(ft))
            pt = np.zeros(shape=data_cube.shape[2], dtype='float32')
            for k in range(data_cube.shape[2]):
                if ft[k]>=0:
                    pt[k] = np.arctan(gt[k] / ft[k])
                elif ft[k]<0:
                    if gt[k]>=0:
                        pt[k] = np.arctan(gt[k] / ft[k]) + np.pi
                    elif gt[k]<0:
                        pt[k] = np.arctan(gt[k] / ft[k]) - np.pi
            output[i, j, :] = pt
            print(i, j)
    return output
if __name__ == '__main__':
    print('hello!')
    data_cube = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
    data_cube = np.moveaxis(data_cube, 0, -1)
    data_cube = data_cube[0:1024, :392, 21:373]
    # spectrum(data_cube[:, 121, :])
    # Gaussian_kernel = Gaussain_kernel_generate(10)
    # spectrum(input_data_norm(data_cube[:, 121, :]), Gaussian_kernel)
    # b = np.array([0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071, 0])
    # c = scipy.signal.hilbert(b)
    # print('c', c)
    # d = np.imag(c)
    # e = np.arctan(d / b)
    # e = e/3.14159
    # print('e', e)
    phase = hilbert_transform(data_cube)
    np.save(r"Ins_phase.npy", phase)
