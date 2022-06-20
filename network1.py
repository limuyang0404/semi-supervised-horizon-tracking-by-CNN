# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
def conv_init(layer):
    """
    :param layer: 需要进行参数初始化的模型层
    :return:
    """
    nn.init.kaiming_normal_(layer.weight.data)#将层内参数进行凯明初始化
    if layer.bias is not None:
        layer.bias.data.zero_()#将层内偏置初始化为0
    return
class ResBlock(nn.Module):
    """
    ResBlock为残差模块，用于自编码器和U型网络中以替代传统模型中的连续卷积层+最大池化层结构
    """
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels:模型输入数据的通道数，整型
        :param out_channels: 模型输出的通道数，整型
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, stride=1)#kernel_size卷积核尺寸，padding输入数据外层
        #填充，stride卷积核滑动距离
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=2)
        # self.conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2)
        self.ReLU1 = nn.ReLU(inplace=True)#ReLU激活函数
        self.ReLU2 = nn.ReLU(inplace=True)
        self.ReLU3 = nn.ReLU(inplace=True)
        # self.ReLU4 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(in_channels)#Batch Normalization，括号内为输入的通道数
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.norm4 = nn.BatchNorm2d(out_channels)
        # self.norm5 = nn.BatchNorm2d(out_channels)
        pass
    def forward(self, x):#forward函数是模型的正向传播函数，属于nn.Module父类，当执行nn.Module()时获得的输出为forward的输出，x为输入数据
        conv1 = self.conv1(x)#将__init__函数中的模型各部分依照顺序连接起来，后一层的输入是前一层的输出
        conv1 = self.norm1(conv1)
        ReLU1 = self.ReLU1(conv1)
        conv2 = self.conv2(ReLU1)
        conv2 = self.norm2(conv2)
        ReLU2 = self.ReLU2(conv2)
        conv3 = self.conv3(ReLU2)
        conv3 = self.norm3(conv3)
        conv4 = self.conv4(x)
        conv4 = self.norm4(conv4)
        conv5 = self.ReLU3(conv4 + conv3)
        # conv5 = self.conv5(conv5)
        # conv5 = self.norm5(conv5)
        out = conv5
        return out
    def initialize(self):#对模型的各层进行初始化
        conv_init(self.conv1)
        conv_init(self.conv2)
        conv_init(self.conv3)
        conv_init(self.conv4)
        # conv_init(self.conv5)
class UpSample(nn.Module):
    """
    UpSample为与ResBlock类对应的结构
    """
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 模型输入数据的通道数，整型
        :param out_channels: 模型输出的通道数，整型
        """
        super(UpSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2, stride=1)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.ReLU0 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        pass
    def forward(self, x):
        x = self.ReLU0(x)
        conv1 = self.conv1(x)
        conv1 = self.norm1(conv1)
        ReLU1 = self.ReLU1(conv1)
        conv2 = self.conv2(ReLU1)
        conv2 = self.norm2(conv2)
        conv3 = self.conv3(x)
        conv3 = self.norm3(conv3)
        out = self.ReLU2(conv2 + conv3)
        return out
    def initialize(self):
        conv_init(self.conv1)
        conv_init(self.conv2)
        conv_init(self.conv3)
class DoubleConv(nn.Module):
    """
    连续卷积层结构
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        pass
    def forward(self, x):
        feature1 = self.conv1(x)
        feature1 = self.norm1(feature1)
        feature1 = self.ReLU1(feature1)
        feature2 = self.conv2(feature1)
        feature2 = self.norm2(feature2)
        out = self.ReLU2(feature2)
        return out
    def initialize(self):
        conv_init(self.conv1)
        conv_init(self.conv2)
        pass
class AutoEncoder(nn.Module):
    """
    用于无监督学习的自编码器结构，主要由连续卷积层、残差模块、UpSample模块等构成，期望输出与模型输入相同
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(AutoEncoder, self).__init__()
        self.conv1 = DoubleConv(in_channels=in_channels, out_channels=32)#连续卷积层
        self.conv2 = ResBlock(in_channels=32, out_channels=64)
        self.conv3 = ResBlock(in_channels=64, out_channels=128)
        self.middle = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.ReLU1 = nn.ReLU()
        self.maxunpool1 = nn.Upsample(scale_factor=2, mode='nearest')#最临近插值，将数据在长和宽两个维度都拓展为两倍
        self.batchnorm_mup1 = nn.BatchNorm2d(128)
        self.ReLU_mup1 = nn.ReLU()
        self.conv4 = UpSample(in_channels=128, out_channels=64)
        self.maxunpool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.batchnorm_mup2 = nn.BatchNorm2d(64)
        self.ReLU_mup2 = nn.ReLU()
        self.conv5 = UpSample(in_channels=64, out_channels=32)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
    def forward(self, x):
        feature1 = self.conv1(x)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        feature_middle = self.middle(feature3)
        feature_middle_ba = self.batchnorm1(feature_middle)
        feature_middle_re = self.ReLU1(feature_middle_ba)
        maxunpool1 = self.maxunpool1(feature_middle_re)
        maxunpool1_ba = self.batchnorm_mup1(maxunpool1)
        maxunpool1_re = self.ReLU_mup1(maxunpool1_ba)
        feature4 = self.conv4(maxunpool1_re)
        maxunpool2 = self.maxunpool2(feature4)
        maxunpool2_ba = self.batchnorm_mup2(maxunpool2)
        maxunpool2_re = self.ReLU_mup2(maxunpool2_ba)
        feature5 = self.conv5(maxunpool2_re)
        feature6 = self.conv6(feature5)
        return feature6
    def ParameterInitialize(self):
        self.conv1.initialize()
        self.conv2.initialize()
        self.conv3.initialize()
        self.conv4.initialize()
        self.conv5.initialize()
        conv_init(self.conv6)
        conv_init(self.middle)
        print('All conv have been initialized!')
        pass
class FreezeUnet(nn.Module):
    """
    用于有监督学习的U型网络，主要由连续卷积层、残差模块、上采样模块等构成，带有跳跃连接结构，模型最初的层载入自编码器权重后无法被更新
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(FreezeUnet, self).__init__()
        self.conv1 = DoubleConv(in_channels=in_channels, out_channels=32)
        self.conv2 = ResBlock(in_channels=32, out_channels=64)
        self.conv3 = ResBlock(in_channels=64, out_channels=128)
        self.conv4 = ResBlock(in_channels=128, out_channels=256)
        self.conv5 = ResBlock(in_channels=256, out_channels=512)
        self.middle = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.batchnorm_middle = nn.BatchNorm2d(512)
        self.ReLU_middle = nn.ReLU()
        self.maxunpool1 = nn.Upsample(scale_factor=2, mode='nearest')#mode = 'bilinear'?
        self.batchnorm_mup1 = nn.BatchNorm2d(512)
        self.ReLU_mup1 = nn.ReLU()
        self.conv6 = UpSample(in_channels=512, out_channels=256)
        self.maxunpool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.batchnorm_mup2 = nn.BatchNorm2d(256)
        self.ReLU_mup2 = nn.ReLU()
        self.conv7 = UpSample(in_channels=256, out_channels=128)
        self.maxunpool3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.batchnorm_mup3 = nn.BatchNorm2d(128)
        self.ReLU_mup3 = nn.ReLU()
        self.conv8 = UpSample(in_channels=128, out_channels=64)
        self.maxunpool4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.batchnorm_mup4 = nn.BatchNorm2d(64)
        self.ReLU_mup4 = nn.ReLU()
        self.conv9 = UpSample(in_channels=64, out_channels=32)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.batchnorm10 = nn.BatchNorm2d(32)
        self.ReLU10 = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.batchnorm11 = nn.BatchNorm2d(32)
        self.ReLU11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, padding=0, stride=1)

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.batchnorm13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.batchnorm14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.batchnorm15 = nn.BatchNorm2d(64)
        self.conv16 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, stride=1)
        self.batchnorm16 = nn.BatchNorm2d(32)
    def forward(self, x):
        with torch.no_grad():#禁止第一层参数随训练变化
            feature1 = self.conv1(x)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        feature4 = self.conv4(feature3)
        feature5 = self.conv5(feature4)
        feature_middle = self.middle(feature5)
        feature_middle_ba = self.batchnorm_middle(feature_middle)
        feature_middle_re = self.ReLU_middle(feature_middle_ba)
        unpool1 = self.maxunpool1(feature_middle_re)
        unpool1_ba = self.batchnorm_mup1(unpool1)
        unpool1_re = self.ReLU_mup1(unpool1_ba)
        feature6 = self.conv6(unpool1_re)
        cat1 = self.conv13(feature4)
        cat1_ba = self.batchnorm13(cat1)
        # temp1 = torch.cat((cat1_ba, unpool1), dim=1)
        temp1 = feature6 + cat1_ba#特征拼接
        # temp1.backward()
        unpool2 = self.maxunpool2(temp1)
        unpool2_ba = self.batchnorm_mup2(unpool2)
        unpool2_re = self.ReLU_mup2(unpool2_ba)
        feature7 = self.conv7(unpool2_re)
        cat2 = self.conv14(feature3)
        cat2_ba = self.batchnorm14(cat2)
        temp2 = feature7 + cat2_ba
        # temp2.backward()
        unpool3 = self.maxunpool3(temp2)
        unpool3_ba = self.batchnorm_mup3(unpool3)
        unpool3_re = self.ReLU_mup3(unpool3_ba)
        feature8 = self.conv8(unpool3_re)
        cat3 = self.conv15(feature2)
        cat3_ba = self.batchnorm15(cat3)
        temp3 = feature8 + cat3_ba
        # temp3.backward()
        unpool4 = self.maxunpool4(temp3)
        unpool4_ba = self.batchnorm_mup4(unpool4)
        unpool4_re = self.ReLU_mup4(unpool4_ba)
        feature9 = self.conv9(unpool4_re)
        cat4 = self.conv16(feature1)
        cat4_ba = self.batchnorm16(cat4)
        temp4 = feature9 + cat4_ba
        # temp4.backward()
        feature10 = self.conv10(temp4)
        feature10_ba = self.batchnorm10(feature10)
        feature10_re = self.ReLU10(feature10_ba)
        feature11 = self.conv11(feature10_re)
        feature11_ba = self.batchnorm11(feature11)
        feature11_re = self.ReLU10(feature11_ba)
        feature12 = self.conv12(feature11_re)
        return feature12
    def ParameterInitialize(self):
        self.conv1.initialize()
        self.conv2.initialize()
        self.conv3.initialize()
        self.conv4.initialize()
        self.conv5.initialize()
        self.conv6.initialize()
        self.conv7.initialize()
        self.conv8.initialize()
        self.conv9.initialize()
        conv_init(self.middle)
        conv_init(self.conv10)
        conv_init(self.conv11)
        conv_init(self.conv12)
        conv_init(self.conv13)
        conv_init(self.conv14)
        conv_init(self.conv15)
        conv_init(self.conv16)
        print('All conv have been initialized!')
if __name__ == "__main__":
    print('hello')
    x1 = torch.rand(size=(1, 1, 128, 128))
    # net = ResBlock(8, 2)
    # net = AutoEncoder(1, 1)
    net = FreezeUnet(1, 6)
    # net = AutoEncoder(1, 1)
    # net.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # net.initialize()
    net.ParameterInitialize()
    total_params = sum(p.numel() for p in net.parameters())
    print('parameter number:\n', total_params)
    # print('net dir:\n', dir(net))
    # print('The net:', net)
    # print('*'*60)
    # print('The state_dict:', net.state_dict())
    net.train()
    # a = nn.Upsample(scale_factor=2, mode='nearest')
    # b = a(x1)
    # x1 = x1.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    output = net(x1)
    print(type(output))
    print(output.size())
    # input = torch.tensor([[[[1., 2, 3, 4],
    #                         [5, 6, 7, 8],
    #                         [9, 10, 11, 12],
    #                         [13, 14, 15, 16]]]])
    # pool = nn.MaxPool2d(2, stride=2)
    # a= pool(input)
    # print(a)
    # d = np.array(np.arange(16, 0, -1)).reshape((1, 1, 4, 4))
    # c = torch.tensor(d)
    # print(c, type(c))
    # unpool = nn.Upsample(scale_factor=2, mode='nearest')
    # b = unpool(a)
    # print(b)

    # x2 = torch.rand(size=(1, 1, 4, 4))
    # x1 = np.arange(16)
    # x1 = list(x1)
    # print(x1, type(x1))
    # # x2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).float().reshape((2, 1, 2, 4))
    # x2 = torch.tensor((), dtype=torch.float32)
    # x2 = x2.new_zeros(size=(10, 5, 4, 4))
    # # x2[0, 0, :, :] = torch.tensor(x1).float().reshape((4, 4))
    # x2 = x2 + torch.tensor(x1).float().reshape((4, 4))
    # # x2[1, :, :] = torch.tensor(x1[::-1]).float().reshape((4, 4))
    # print('x2:', type(x2), x2)
    # a = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    # b, c = a(x2)
    # print(b)
    # print('__________')
    # print(c)
    # d = nn.MaxUnpool2d(kernel_size=2, stride=2)
    # e = d(input=b, indices=c)
    # print(e)
    # a = np.load(r"/home/limuyang/New_zealand_data/label_3.npy")
    # print(a.shape)
    # plt.imshow(np.moveaxis(a[742, :, :], 0, -1), cmap='rainbow')
    # plt.show()
