# -*- coding: utf-8 -*-
import torch
import numpy as np
from network1 import AutoEncoder, FreezeUnet
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

def random_batch(data_cube, index):
    """
    :param data_cube: 输入的地震数据
    :param index: 索引值，整型
    :return: 用于张量转换的四维数组，第一个维度为批次样本数，第二个维度为通道数，第三第四个维度为样本的道数和采样点数
    """
    output = np.zeros(shape=(1, 1, 1024, 352), dtype='float32')
    data_img = input_data_norm(data_cube[:, index, :])#输入数据标准化
    output[0, 0, :, :] = data_img
    return output
def input_data_norm(input_cube_img):
    """
    :param input_cube_img: 待标准化的二维数组
    :return: 标准化后的二维数组
    """
    mean1 = np.mean(input_cube_img)#均值
    deviation1 = np.var(input_cube_img) ** 0.5#标准差
    input_cube_img_norm = (input_cube_img - mean1) / deviation1#A_norm = (A - A_mean) / A_dev
    return input_cube_img_norm
def model_predict_result(model_output):
    """
    :param model_output: 模型的实际输出，四维张量，第一维为样本在批次的索引，第二维为通道数，第三第四维为样本的道数和采样点数
    :return: 模型输出在通道维度上最大值的通道值索引，对应了模型给出的预测概率的最大类别
    """
    predict_result = torch.nn.Softmax(dim=1)(model_output)#在通道维度求取SoftMax值
    class_pos, class_no = torch.max(predict_result, 1, keepdim=True)#class_pos为通道维度的最大值，class_no为最大值的通道数
    model_output_index = torch.squeeze(class_no).cpu().detach().numpy()#将通道数以多维数组的形式输出
    return model_output_index
def predict_whole_cube_AE(data_file, model_para_file):
    """
    利用自编码器对地震数据进行预测以检测无监督学习效果
    :param data_file: 需要进行预测的地震数据，三维数组
    :param model_para_file: 训练后保存的模型参数
    :return:
    """
    seismic_data = np.load(data_file)
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data = seismic_data[:1024, :384, 21:373]#对输入数据进行裁切以满足训练需求
    print('seismic_data.shape:', seismic_data.shape)
    torch.cuda.empty_cache()#清空缓存
    network = AutoEncoder(in_channels=1, out_channels=1)#自编码器实例化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(str(torch.cuda.device_count()) + ' cards!')
        network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
    network.to(device)
    network.eval()#将模型模式调整为eval()。与train()不同，模型内各参数将无法被更改
    model_path = model_para_file
    network.load_state_dict(torch.load(model_path))#载入模型参数
    model_predict = np.zeros(shape=seismic_data.shape, dtype='float32')
    for i in range(seismic_data.shape[1]):#分剖面预测
        data_input = random_batch(seismic_data, i)
        data_input = (torch.autograd.Variable(torch.Tensor(data_input).float())).to(device)
        model_output = network(data_input)
        a = torch.squeeze(model_output).cpu().detach().numpy()
        model_predict[:, i, :] = a
        print('\rThe %d inline img!'%i)
    print("model_predict shape", model_predict.shape)
    np.save(r"predictAE.npy", model_predict)#保存预测结果
    return
def predict_whole_cube_FreezeUnet(data_file, model_para_file):
    """
    使用训练后的含残差模块U型网络进行目的层位对应地层序列的预测
    :param data_file: 需要进行预测的地震数据，三维数组
    :param model_para_file: 训练后保存的模型参数
    :return:
    """
    seismic_data = np.load(data_file)
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data = seismic_data[:1024, :384, 21:373]
    torch.cuda.empty_cache()
    network = FreezeUnet(in_channels=1, out_channels=6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(str(torch.cuda.device_count()) + ' cards!')
        network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
    network.to(device)
    network.eval()
    model_path = model_para_file
    network.load_state_dict(torch.load(model_path))
    model_predict = np.zeros(shape=seismic_data.shape, dtype='float32')
    for i in range(seismic_data.shape[1]):
        data_input = random_batch(seismic_data, i)
        data_input = (torch.autograd.Variable(torch.Tensor(data_input).float())).to(device)
        model_output = network(data_input)
        predict_whole = model_predict_result(model_output=model_output)
        model_predict[:, i, :] = predict_whole
        print('\rThe %d inline img!'%i, np.min(predict_whole), np.max(predict_whole))
    np.save(r"predictFreeze8_100.npy", model_predict)
    return
if __name__ == "__main__":
    print('hello')
    predict_whole_cube_AE(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"saved_modelAE22.pt")
    # predict_whole_cube_FreezeUnet(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"saved_modelFreeze8_1.pt")
