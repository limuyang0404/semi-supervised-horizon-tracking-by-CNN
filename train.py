# -*- coding: utf-8 -*-
import numpy as np
from network1 import AutoEncoder, FreezeUnet
import torch
from torch import nn
from collections import Counter
import pandas as pd
import segyio
import matplotlib.pyplot as plt
from random import shuffle
def random_batch_AE(data_cube, batch_list_input, z):
    """
    :param data_cube: 用于训练自编码器的地震数据，三维数组
    :param batch_list_input: 用于确定样本在地震数据位置的索引，列表
    :param z: 模型训练的当前批次数，整型
    :return: 用于训练自编码器的随机批次样本，四维数组
    """
    whole_index = batch_list_input * 1#将传入的索引列表复制一份避免更改原列表
    whole_index.extend(batch_list_input)
    whole_index.extend(batch_list_input)#将列表复制两次进行扩充
    whole_index = whole_index[z*100:z*100+100]#将列表切出长度为100的切片
    data_output = np.zeros(shape=(100, 1, 384, 352), dtype='float32')#输出的随机批次样本，其中100为批次内的样本数量，1为通道数，384和352分别为样本的道数和采样点数
    new_img = np.zeros(shape=(384, 352), dtype='float32')#随机批次内的样本
    for i in range(100):#每个批次有100个样本
        # a = np.random.randint(0, 8)
        img_type = whole_index[i][0]#索引对应的样本类型，0为xline样本，1为inline样本
        img_index = whole_index[i][1]#索引对杨的样本在原剖面的位置
        if img_type == 0:
            data_img = input_data_norm(data_cube[img_index, :, :])#样本标准化
            if whole_index[i][3] == 0:#0-7对应8种不同的变换用于实现数据增强
                data_output[i, 0, :, :] = data_img
            elif whole_index[i][3] == 1:
                data_output[i, 0, :, :] = data_img[::-1, :]
            elif whole_index[i][3] == 2:
                data_output[i, 0, :, :] = data_img[:, ::-1]
            elif whole_index[i][3] == 3:
                data_output[i, 0, :, :] = data_img[::-1, ::-1]
            elif whole_index[i][3] == 4:
                a = data_img[:192, :]
                new_img[:192, :] = a
                new_img[192:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
            elif whole_index[i][3] == 5:
                a = data_img[:192, :]
                new_img[192:, :] = a
                new_img[:192, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
            elif whole_index[i][3] == 6:
                a = data_img[192:, :]
                new_img[:192, :] = a
                new_img[192:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
            elif whole_index[i][3] == 7:
                a = data_img[192:, :]
                new_img[192:, :] = a
                new_img[:192, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
        elif img_type == 1:
            index_start = whole_index[i][2]
            data_img = input_data_norm(data_cube[320*index_start:320*index_start+384, img_index, :])#样本标准化
            if whole_index[i][3] == 0:
                data_output[i, 0, :, :] = data_img
            elif whole_index[i][3] == 1:
                data_output[i, 0, :, :] = data_img[::-1, :]
            elif whole_index[i][3] == 2:
                data_output[i, 0, :, :] = data_img[:, ::-1]
            elif whole_index[i][3] == 3:
                data_output[i, 0, :, :] = data_img[::-1, ::-1]
            elif whole_index[i][3] == 4:
                a = data_img[:192, :]
                new_img[:192, :] = a
                new_img[192:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
            elif whole_index[i][3] == 5:
                a = data_img[:192, :]
                new_img[192:, :] = a
                new_img[:192, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
            elif whole_index[i][3] == 6:
                a = data_img[192:, :]
                new_img[:192, :] = a
                new_img[192:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
            elif whole_index[i][3] == 7:
                a = data_img[192:, :]
                new_img[192:, :] = a
                new_img[:192, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
    return data_output
def random_batch_FreezeUnet(data_cube, label_cube, batch_list_input):
    """
    :param data_cube: 用于训练冻结参数的U型网络的地震数据，三维数组
    :param label_cube: 用于训练冻结参数的U型网络的地层序列标签，三维数组
    :param batch_list_input: 用于确定样本在地震数据位置的索引，列表
    :return: 用于训练冻结参数的U型网络的随机批次样本，振幅四维数组/地层序列三维数组
    """
    whole_index = batch_list_input * 1
    whole_index.extend(batch_list_input)
    shuffle(whole_index)
    data_output = np.zeros(shape=(56, 1, 384, 352), dtype='float32')#输出的随机批次样本，其中56为批次内的样本数量，1为通道数，384和352分别为样本的道数和采样点数
    label_output = np.zeros(shape=(56, 384, 352), dtype='float32')#相较于输入数据，标签没有通道维度，通过后续独热编码实现匹配
    new_img = np.zeros(shape=(384, 352), dtype='float32')
    for i in range(56):
        img_type = whole_index[i][0]
        img_index = whole_index[i][1]
        if img_type == 0:
            data_img = input_data_norm(data_cube[img_index, :, :])
            label_img = label_cube[img_index, :, :]
            if whole_index[i][3] == 0:
                data_output[i, 0, :, :] = data_img
                label_output[i, :, :] = label_img
            elif whole_index[i][3] == 1:
                data_output[i, 0, :, :] = data_img[::-1, :]
                label_output[i, :, :] = label_img[::-1, :]
            elif whole_index[i][3] == 2:
                data_output[i, 0, :, :] = data_img[:, ::-1]
                label_output[i, :, :] = label_img[:, ::-1]
            elif whole_index[i][3] == 3:
                data_output[i, 0, :, :] = data_img[::-1, ::-1]
                label_output[i, :, :] = label_img[::-1, ::-1]
            elif whole_index[i][3] == 4:
                a = data_img[:192, :]
                new_img[:192, :] = a
                new_img[192:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[:192, :]
                new_img[:192, :] = b
                new_img[192:, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 5:
                a = data_img[:192, :]
                new_img[192:, :] = a
                new_img[:192, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[:192, :]
                new_img[192:, :] = b
                new_img[:192, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 6:
                a = data_img[192:, :]
                new_img[:192, :] = a
                new_img[192:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[192:, :]
                new_img[:192, :] = b
                new_img[192:, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 7:
                a = data_img[192:, :]
                new_img[192:, :] = a
                new_img[:192, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[192:, :]
                new_img[192:, :] = b
                new_img[:192, :] = b[::-1, :]
                label_output[i, :, :] = new_img
        elif img_type == 1:
            index_start = np.random.randint(3)
            data_img = input_data_norm(data_cube[320*index_start:320*index_start+384, img_index, :])
            label_img = label_cube[320*index_start:320*index_start+384, img_index, :]
            if whole_index[i][3] == 0:
                data_output[i, 0, :, :] = data_img
                label_output[i, :, :] = label_img
            elif whole_index[i][3] == 1:
                data_output[i, 0, :, :] = data_img[::-1, :]
                label_output[i, :, :] = label_img[::-1, :]
            elif whole_index[i][3] == 2:
                data_output[i, 0, :, :] = data_img[:, ::-1]
                label_output[i, :, :] = label_img[:, ::-1]
            elif whole_index[i][3] == 3:
                data_output[i, 0, :, :] = data_img[::-1, ::-1]
                label_output[i, :, :] = label_img[::-1, ::-1]
            elif whole_index[i][3] == 4:
                a = data_img[:192, :]
                new_img[:192, :] = a
                new_img[192:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[:192, :]
                new_img[:192, :] = b
                new_img[192:, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 5:
                a = data_img[:192, :]
                new_img[192:, :] = a
                new_img[:192, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[:192, :]
                new_img[192:, :] = b
                new_img[:192, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 6:
                a = data_img[192:, :]
                new_img[:192, :] = a
                new_img[192:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[192:, :]
                new_img[:192, :] = b
                new_img[192:, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 7:
                a = data_img[192:, :]
                new_img[192:, :] = a
                new_img[:192, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[192:, :]
                new_img[192:, :] = b
                new_img[:192, :] = b[::-1, :]
                label_output[i, :, :] = new_img
    return data_output, label_output
def class_point_number(label_cube):
    """
    :param label_cube:人工标签，三维数组
    :return:
    """
    inline_list = [46, 196, 346]
    xline_list = [142, 342, 542, 742, 942]
    img_range = 392
    class_point_counter = Counter()#空的Counter字典
    class_point_counter1 = Counter()
    for i in range(3):
        for j in range(3):
            class_point_counter += Counter(label_cube[320*j:320*j+384, i*150+46, :].flatten())#inline剖面样本
        print("class_point_counter:", class_point_counter, i)#人工标签中各个类别样点的数量统计
    for i in range(5):
        class_point_counter1 += Counter(label_cube[i*200+142, :, :].flatten())#xline剖面样本
        print("class_point_counter1:", class_point_counter1, i)
    return
def input_data_norm(input_cube_img):
    """
    :param input_cube_img:输入数据，二维数组
    :return: 标准化后的输入数据
    """
    mean1 = np.mean(input_cube_img)#求均值
    deviation1 = np.var(input_cube_img) ** 0.5#求标准差
    input_cube_img_norm = (input_cube_img - mean1) / deviation1#标准化
    return input_cube_img_norm
def model_train(mode):
    """
    :param mode: 训练模式，AE代表自编码器无监督学习，Freeze代表冻结参数的U型网络有监督学习
    :return:
    """
    if mode == 'AE':
        print('hello')
        torch.cuda.empty_cache()#清空显存的缓存
        network = AutoEncoder(in_channels=1, out_channels=1)#自编码器类的实例化
        network.ParameterInitialize()#自编码器参数初始化
        total_params = sum(p.numel() for p in network.parameters())#模型参数的总数量
        print('parameter number:\n', total_params)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#设置torch.device，如果有可用的显卡为"cuda:0"，如果没有则为cpu
        MSE = nn.MSELoss(reduction='mean')#均方误差函数作为损失函数
        if torch.cuda.device_count() > 1:
            print(str(torch.cuda.device_count()) + ' cards!')
            network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])#适用于172.19.144.52工作站的四张卡
        network.to(device)#将模型送入显卡
        network.train()#模型的模式调整为训练
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)#Adam优化器，初始学习率为0.001
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975, last_epoch=-1)#指数型学习率调整，每隔step_size更新一次学习率，每次为上次的gamma倍
        for state in optimizer.state.values():
            for k, v in state.items():  # for k, v in d.items()  Iterate the key and value simultaneously
                if torch.is_tensor(v):
                    state[k] = v.cuda()#将优化器送入显卡
                pass
            pass
        loss_list = []#损失列表
        data_cube = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")#使用的地震数据文件路径
        data_cube = np.moveaxis(data_cube, 0, -1)
        data_cube = data_cube[:1024, :384, 21:373]#地震数据切片，下采样可能会限制输入数据的尺寸
        batch_list1 = []#索引列表
        for i in range(1024):
            for j in range(1):
                for k in range(8):
                    batch_list1.append((0, i, j, k))
        for i in range(384):
            for j in range(3):
                for k in range(8):
                    batch_list1.append((1, i, j, k))
        for z in range(2):#20个epoch
            shuffle(batch_list1)#打乱索引列表
            for i in range(3):#350个batch
                torch.cuda.empty_cache()##清空显存的缓存
                data = random_batch_AE(data_cube=data_cube, batch_list_input=batch_list1, z=i)#获取随机批次
                data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)#转换为浮点型张量
                # label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)
                # print(network.state_dict()['module.conv1.weight'])
                output = network(data)#模型输出
                # label1, output1 = label_mask_and_select(label, output)
                # loss = MSE(output, label)
                loss = MSE(output, data)#损失计算
                print(r"The %dth epoch's %dth batch's loss is:" % (z, i + 1), loss)
                loss_list.append(loss.cpu())
                loss.backward()#损失反向传递
                optimizer.step()#模型经优化器进行参数更新
                optimizer.zero_grad()#损失梯度清零
            scheduler.step()#学习率调整
            if (z + 1) % 1 == 0:
                torch.save(network.state_dict(),
                           'saved_modelAE2' + str(int((z + 1) // 1)) + '.pt')  # 网络保存为saved_modelAE2.pt
                torch.save(optimizer.state_dict(), 'optimizer' + '.pth')
        print('len(loss):', len(loss_list))
        np.savetxt(r'loss_valueAE2.txt', torch.Tensor(loss_list).detach().numpy())#模型损失
    elif mode == 'Freeze':
        print('hello')
        torch.cuda.empty_cache()#清空显存的缓存
        network = FreezeUnet(in_channels=1, out_channels=6)#U型网络类实例化
        network.ParameterInitialize()
        total_params = sum(p.numel() for p in network.parameters())
        print('parameter number:\n', total_params)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.4410959118539384, 0.7977417345271601, 1.0732173989882963, 0.7387858454447427, 0.47772838894573333, 0.47143072024012955, 1]).to(device),ignore_index=-1)
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.0053348416808006675, 1.1794575289895175, 1.1865171710991096, 1.1917482672384678, 1.2016214055722816, 1.2353207854198223]).to(device),ignore_index=-1)
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9380593817231722, 0.3016548322264485, 0.40692383472537275, 0.2834924401887393, 0.18573017749836984, 0.18646788475400142, 4.697671448883896]).to(device),ignore_index=-1)
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.765683352262653, 0.2483766316565284, 0.33474340556961707, 0.23163537767618422, 0.1506609242891989, 0.15104440805177177, 5.117855900494046]).to(device),ignore_index=-1)
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.4457178827949524, 0.7959277962480626, 1.070451779007012, 0.7374964055930279, 0.47693183812448464, 0.4734742982324609]).to(device),ignore_index=-1)
        cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.4454488213472736, 0.7946545357352693, 1.070625020362341, 0.7387402142882846, 0.47806968436028613, 0.47246172390654545]).to(device),ignore_index=-1)
        #带权重交叉熵函数作为损失函数
        if torch.cuda.device_count() > 1:
            print(str(torch.cuda.device_count()) + ' cards!')
            network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        network.to(device)
        network.train()
        model_path = 'saved_modelAE22.pt'#载入自编码器训练后的模型参数
        pre_weights = torch.load(model_path)
        print(pre_weights.keys())
        print('*'*70)
        # print(pre_weights['left_conv_1.conv_ReLU.1.running_mean'])
        del_key_list = []
        # del_keyword = ["module.conv2", "conv3", "conv4", "middle", "conv5", "conv6", "conv7", "conv8", "conv9", "module.batchnorm"]
        del_keyword = ["module.conv4", "module.middle", "module.conv5", "module.conv6", "module.batchnorm"]#自编码器中不用于参数迁移的层
        for key, _ in pre_weights.items():
            for i in range(len(del_keyword)):
                if del_keyword[i] in key:
                    del_key_list.append(key)
            # elif '_3' in key:
            #     del_key_list.append(key)
        for key in del_key_list:
            del(pre_weights[key])
        # print(pre_weights.keys())
        print(pre_weights.keys())
        missing_keys, unexpected_keys = network.load_state_dict(pre_weights, strict=False)#自编码器参数加载
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)#Adam优化器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95, last_epoch=-1)#1, 0.975, -1
        for state in optimizer.state.values():
            for k, v in state.items():  # for k, v in d.items()  Iterate the key and value simultaneously
                if torch.is_tensor(v):
                    state[k] = v.cuda()
                pass
            pass
        loss_list = []
        data_cube = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")#训练用地震数据
        data_cube = np.moveaxis(data_cube, 0, -1)
        data_cube = data_cube[:1024, :384, 21:373]
        # data_cube_mean = np.mean(data_cube)
        # data_cube_deviation = np.var(data_cube)
        # data_cube = (data_cube - data_cube_mean) / data_cube_deviation
        # data_cube_cut = np.zeros(shape=(data_cube.shape[0], 3, data_cube.shape[2]), dtype='float32')
        label_cube = np.load(r"/home/limuyang/New_zealand_data/label_3.npy")#训练用地层序列标签
        label_cube = label_cube[0:1024, :384, 21:373]
        print('label_cube shape:', label_cube.shape)
        class_point_number(label_cube)#标签样点数量统计
        # ins_phase = np.load(r"Ins_phase.npy")

        # seismic_cube_cut = np.zeros(shape=(data_cube.shape[0], 3, data_cube.shape[2]), dtype='float32')
        # label_cube_cut = np.zeros(shape=(data_cube.shape[0], 3, data_cube.shape[2]), dtype='float32')
        batch_list1 = []#索引列表
        for i in range(5):
            for j in range(1):
                for k in range(2):
                    batch_list1.append((0, i*200+142, j, k))
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    batch_list1.append((1, i*150+46, j, k))
        # for i in range(3):
        #     for j in range(8):
        #         batch_list1.append((1, i*150+46, j))
        # z_index = np.zeros(shape=(256, 352), dtype='float32')
        # for i in range(352):
        #     z_index[:, i] = i * 0.01 - 1.26
        # print('Counter of label: ', Counter(label_cube.flatten()))
        # label_count(label_cube_cut)
        for z in range(100):#epoch
            shuffle(batch_list1)
            for i in range(60):#批次
                # print("module.conv1.conv1.weight:", network.state_dict()['module.conv1.conv1.weight'])
                # print("module.conv8.conv1.weight:", network.state_dict()['module.conv8.conv1.weight'])
                torch.cuda.empty_cache()
                data, label = random_batch_FreezeUnet(data_cube=data_cube, label_cube=label_cube, batch_list_input=batch_list1)
                data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
                label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)#张量需转化为long型张量
                # print(network.state_dict()['module.conv4.conv3.weight'])
                output = network(data)
                # if z%1 == 0 and i % 59 == 0 and i>0:
                #     plt.cla()
                #     plt.clf()
                #     aba = torch.nn.Softmax(dim=1)(output)
                #     class_pos, class_no = torch.max(aba, 1, keepdim=True)
                #     model_output_index = torch.squeeze(class_no).cpu().detach().numpy()
                #     plt.imshow(model_output_index[0, :, :])
                #     plt.show()
                # label1, output1 = label_mask_and_select(label, output)
                # loss = MSE(output, label)
                loss = cross_entropy(output, label)
                print(r"The %dth epoch's %dth batch's loss is:" % (z, i + 1), loss)
                loss_list.append(loss.cpu())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            if (z + 1) % 1 == 0:
                torch.save(network.state_dict(),
                           'saved_modelFreeze8_' + str(int((z + 1) // 1)) + '.pt')  # 网络保存为saved_model.pt
                torch.save(optimizer.state_dict(), 'optimizer' + '.pth')
        print('len(loss):', len(loss_list))
        np.savetxt(r'loss_valueFreeze8.txt', torch.Tensor(loss_list).detach().numpy())
    return
if __name__ == '__main__':
    print('Start!')
    model_train(mode='Freeze')
    # model_train(mode='AE')
    # a = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
    # print(a.shape)