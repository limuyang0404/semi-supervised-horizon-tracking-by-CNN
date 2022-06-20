# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as colors

def check_inline(seismic_file, label_file, predict_file, inline_index):
    seismic_data = np.load(seismic_file)
    print('seismic_data shape:', seismic_data.shape)
    seismic_data_mean = np.mean(seismic_data)
    seismic_data_deviation = np.var(seismic_data)
    seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = (seismic_data - seismic_data_mean) / seismic_data_deviation
    # seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data = seismic_data[0:1024, inline_index, 21:373]
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    label_data = np.load(label_file)
    print('label_data shape:', label_data.shape)
    label_data = label_data[0:1024, inline_index, 21:373]
    label_data = np.moveaxis(label_data, 0, -1)
    predict_data = np.load(predict_file)
    print('predict_data shape:', predict_data.shape)
    predict_data = predict_data[:, inline_index, :]
    predict_data = np.moveaxis(predict_data, 0, -1)
    alpha1 = np.ones(shape=label_data.shape, dtype='float32')
    alpha2 = np.ones(shape=label_data.shape, dtype='float32')
    alpha1[np.where(label_data<1)] = 0
    # alpha2[np.where(predict_data<1)] = 0
    # for i in range(479):
    #     for j in range(1024):
    #         if predict_data[i, j] != predict_data[i+1, j]:
    #             alpha2[i, j] = 1
    # plt.subplot(1, 2, 1)
    # plt.title('Label', fontsize=18)
    # plt.imshow(seismic_data, cmap=plt.cm.gray)
    # plt.imshow(label_data, alpha=alpha1)
    # plt.subplot(1, 2, 2)
    plt.title('Predict', fontsize=18)
    # plt.imshow(seismic_data, cmap=plt.cm.gray)
    plt.imshow(predict_data, cmap=plt.cm.gray)
    plt.show()
    # plt.savefig("Label_46.png", dpi=300)
    return
def label_img_edit(img_path_list, label_file):
    label_cube = np.load(label_file)
    print('label_cube shape', label_cube.shape)
    print(label_cube.shape)
    for i in range(len(img_path_list)):
        iline_img = np.zeros(shape=(1043, 501))
        img_array = plt.imread(img_path_list[i])
        img_array = np.moveaxis(img_array, 0, 1)
        print('img_array shape:', img_array.shape)
        img_array = img_array * 255
        for j in range(1043):
            for k in range(501):
                if img_array[j, k, 0] == 64 and img_array[j, k, 1] == 67 and img_array[j, k, 2] == 135:
                    iline_img[j, k] = 1
                elif img_array[j, k, 0] == 41 and img_array[j, k, 1] == 120 and img_array[
                    j, k, 2] == 142:
                    iline_img[j, k] = 2
                elif img_array[j, k, 0] == 34 and img_array[j, k, 1] == 167 and img_array[
                    j, k, 2] == 132:
                    iline_img[j, k] = 3
                elif img_array[j, k, 0] == 121 and img_array[j, k, 1] == 209 and img_array[
                    j, k, 2] == 81:
                    iline_img[j, k] = 4
                elif img_array[j, k, 0] == 253 and img_array[j, k, 1] == 231 and img_array[
                    j, k, 2] == 36:
                    iline_img[j, k] = 5
        label_cube[:, i * 150 + 46, :] = iline_img
        # plt.show(img_array)
    np.save("label_2.npy", label_cube)

    return
def seismic_cube_cut(seismic_file):
    seismic_cube = np.load(seismic_file)
    seismic_cube = np.moveaxis(seismic_cube, 0, -1)
    print(seismic_cube.shape)
    seismic_cube = seismic_cube[:, :, :]
    np.save("seismic_2.npy", seismic_cube)
    return
# def check_inline(seismic_file, label_file, predict_file, inline_index):
#     seismic_data = np.load(seismic_file)
#     seismic_data = seismic_data[:, inline_index, :]
#     seismic_data = np.moveaxis(seismic_data, 0, -1)
#     label_data = np.load(label_file)
#     label_data = label_data[:, inline_index, :]
#     label_data = np.moveaxis(label_data, 0, -1)
#     predict_data = np.load(predict_file)
#     predict_data = predict_data[:, inline_index, :]
#     predict_data = np.moveaxis(predict_data, 0, -1)
#     alpha1 = np.ones(shape=label_data.shape, dtype='float32')
#     alpha2 = np.ones(shape=label_data.shape, dtype='float32')
#     alpha1[np.where(label_data<1)] = 0
#     alpha2[np.where(predict_data<1)] = 0
#     plt.subplot(1, 2, 1)
#     plt.title('Label', fontsize=18)
#     plt.imshow(seismic_data, cmap=plt.cm.gray)
#     plt.imshow(label_data, alpha=alpha1)
#     plt.subplot(1, 2, 2)
#     plt.title('Predict', fontsize=18)
#     plt.imshow(seismic_data, cmap=plt.cm.gray)
#     plt.imshow(predict_data, alpha=alpha2)
#     plt.show()
#     return
def line_horizon_trans_block(label_file):
    """
    将人工解释的层位转化为训练使用的地层序列标签
    :param label_file: 目的层位人工解释结果
    :return:
    """
    label_cube = np.load(label_file)#shape = (1043, 396, 501)
    # print("label_cube's shape", label_cube.shape)
    label_cube[689, 196, 183] = 4
    label_cube[742, 18, 37] = 1
    label_cube[942, 58, 37] = 1#对人工解释结果的部分修改，各目的层位和与其相邻的断层线应包含所有的道号，使得地层序列的边界为目的层位和断层线
    for i in range(1043):
        for j in range(396):
            trace_array = label_cube[i, j, :]
            trace_array = trace_trans(trace_array=trace_array)#对所有道进行遍历，从上到下划分为若干地层序列
            label_cube[i, j, :] = trace_array
            print(i, j)
    np.save(r"label_3.npy", label_cube)#转换后的地层序列标签
    return
def block_add_fault(label_file):
    """
    将断层视作一种特殊的地层序列
    :param label_file: 人工解释结果
    :return:
    """
    label_cube = np.load(label_file)
    seismic_data = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
    print('seismic_data shape:', seismic_data.shape)
    seismic_data_mean = np.mean(seismic_data)
    seismic_data_deviation = np.var(seismic_data)
    seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = (seismic_data - seismic_data_mean) / seismic_data_deviation
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    print('seismic_data shape:', seismic_data.shape)
    seismic_data1 = seismic_data[342, :, :]
    seismic_data1 = np.moveaxis(seismic_data1, 0, -1)
    print('seismic_data shape:', seismic_data1.shape)
    print("label_cube's shape:", label_cube.shape)
    label_data = label_cube[142, :, :]
    label_data = np.moveaxis(label_data, 0, -1)
    fault_xline = np.load(r"Labels/fault/Fault_5Xline.npy")#xline剖面断层解释
    fault_inline = np.load(r"Labels/fault/Fault_3Inline.npy")#inline剖面断层解释
    print("fault_inline's shape:", fault_inline.shape)
    print("fault_xline's shape:", fault_xline.shape)
    alpha1 = np.ones(shape=(501, 396), dtype='float32')
    alpha2 = np.ones(shape=(501, 396), dtype='float32')
    # alpha1[np.where(fault_xline[:, 0, :] < 1)] = 0
    alpha2[np.where(label_data < 1)] = 0
    # plt.imshow(seismic_data1, cmap=plt.cm.gray)
    # # plt.imshow(label_data, alpha=alpha2)
    # plt.imshow(fault_xline[:, 1, :])
    # plt.show()
    fault_cube = np.zeros(shape=(1043, 396, 501), dtype='float32')

    for i in range(3):
        fault_cube[:, i*150+46, :] = np.moveaxis(fault_inline[:, :, i], 0, -1)
    for i in range(5):
        fault_cube[i*200+142, :, :] = np.moveaxis(fault_xline[:, i, :], 0, -1)
    label_cube[np.where(fault_cube > 0)] = 6
    # label_cube = label_cube + fault_cube
    plt.imshow(seismic_data1, cmap=plt.cm.gray)
    # plt.imshow(label_data, alpha=alpha2)
    plt.imshow(np.moveaxis(label_cube[942, :, :], 0, -1), alpha=alpha1)
    plt.show()
    np.save(r"label_5.npy", label_cube)#将断层视作一种特殊的地层序列后得到的标签
    return
def trace_trans(trace_array):
    """
    对人工解释结果进行转换，输出地层序列
    :param trace_array: 待转换的人工解释结果
    :return:
    """
    output_trace_array = np.zeros(shape=trace_array.shape, dtype='float32')
    horizon_index = []
    for i in range(trace_array.shape[0]):
        if trace_array[i] > 0:
            horizon_index.append([trace_array[i], i])
    if len(horizon_index) > 0:
        for i in range(len(horizon_index)):
            output_trace_array[horizon_index[i][1]:] = horizon_index[i][0]
        # trace_array[:horizon_index[0]] = 0
        # trace_array[horizon_index[0]:horizon_index[1]] = 1
        # trace_array[horizon_index[1]:horizon_index[2]] = 2
        # trace_array[horizon_index[2]:horizon_index[3]] = 3
        # trace_array[horizon_index[3]:horizon_index[4]] = 4
        # trace_array[horizon_index[4]:] = 5
        return output_trace_array
    else:
        return trace_array
def check_inline_block(seismic_file, label_file, predict_file, inline_index):
    """
    显示inline剖面的预测效果
    :param seismic_file: 待预测的地震数据的路径
    :param label_file: 训练使用的标签的路径
    :param predict_file: 预测结果文件的路径
    :param inline_index: 要显示的剖面的索引值，整型
    :return:
    """
    seismic_data = np.load(seismic_file)
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data = seismic_data[:1024, :384, 21:373]
    print("seismic_data shape:", seismic_data.shape)
    # seismic_data = np.moveaxis(seismic_data, 0, -1)
    # print("seismic_data shape:", seismic_data.shape)
    # seismic_data = seismic_data[0:1024, :392, 21:373]
    # print("seismic_data shape:", seismic_data.shape)
    mean = np.mean(seismic_data)
    deviation = np.var(seismic_data) ** 0.5
    # seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = seismic_data[:, inline_index, :]
    print("seismic_data shape!:", seismic_data.shape)
    # seismic_data_mean = np.mean(seismic_data)
    # seismic_data_deviation = np.var(seismic_data)
    # seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = (seismic_data - mean) / deviation
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    print('seismic min', np.min(seismic_data), np.max(seismic_data))
    label_data = np.load(label_file)
    label_data = label_data[:1024, :384, 21:373]
    print("label_data shape:", label_data.shape)
    # label_data = label_data[0:1024, :392, 21:373]
    label_data = label_data[:, inline_index, :]
    label_data = np.moveaxis(label_data, 0, -1)
    predict_data = np.load(predict_file)
    print("predict_data shape:", predict_data.shape)
    # predict_data = predict_data[0:1024, :392, 21:373]
    # predict_data = predict_data[:, :, :608]
    predict_data = predict_data[:, inline_index, :]
    predict_data = np.moveaxis(predict_data, 0, -1)
    print('predict min', np.min(predict_data), np.max(predict_data))
    alpha1 = np.ones(shape=label_data.shape, dtype='float32')
    alpha2 = np.ones(shape=predict_data.shape, dtype='float32')
    alpha1[np.where(label_data < 0)] = 0
    alpha2[np.where(predict_data < 1)] = 0
    blockedge = horizon_line(predict_data)
    # alpha2[np.where(blockedge < 1)] = 0
    print("seismic_data shape:", seismic_data.shape)
    print("predict_data shape:", predict_data.shape)
    plt.subplot(1, 2, 1)#通过pyplot画热图显示，可以通过调整透明度实现叠加显示
    plt.title('Seismic', fontsize=18)
    plt.imshow(seismic_data, cmap=plt.cm.gray)
    # plt.imshow(label_data, cmap=plt.cm.ocean, alpha=alpha1)
    # plt.imshow(label_data, cmap=plt.cm.rainbow)
    plt.subplot(1, 2, 2)
    plt.title('Predict', fontsize=18)
    # plt.imshow(seismic_data, cmap=plt.cm.gray)
    plt.imshow(predict_data, cmap=plt.cm.rainbow)
    # plt.imshow(label_data, cmap=plt.cm.ocean, alpha=alpha1)

    # plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + '0ilineblock.png', predict_data)
    # plt.imshow(blockedge)
    plt.show()
    return
def check_xline_block(seismic_file, label_file, predict_file, xline_index):
    """
    显示xline剖面的预测效果
    :param seismic_file: 待预测地震数据路径
    :param label_file: 训练使用的标签路径
    :param predict_file: 模型预测结果路径
    :param xline_index: 要显示的剖面的索引值，整型
    :return:
    """
    seismic_data = np.load(seismic_file)
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data = seismic_data[:1024, :384, 21:373]
    print("seismic_data shape:", seismic_data.shape)
    # seismic_data = np.moveaxis(seismic_data, 0, -1)
    # print("seismic_data shape:", seismic_data.shape)
    # seismic_data = seismic_data[0:1024, :392, 21:373]
    # print("seismic_data shape:", seismic_data.shape)
    mean = np.mean(seismic_data)
    deviation = np.var(seismic_data) ** 0.5
    # seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = seismic_data[xline_index, :, :]
    print("seismic_data shape!:", seismic_data.shape)
    # seismic_data_mean = np.mean(seismic_data)
    # seismic_data_deviation = np.var(seismic_data)
    # seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = (seismic_data - mean) / deviation
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    print('seismic min', np.min(seismic_data), np.max(seismic_data))
    label_data = np.load(label_file)
    label_data = label_data[:1024, :384, 21:373]
    print("label_data shape:", label_data.shape)
    # label_data = label_data[0:1024, :392, 21:373]
    label_data = label_data[xline_index, :, :]
    label_data = np.moveaxis(label_data, 0, -1)
    predict_data = np.load(predict_file)
    print("predict_data shape:", predict_data.shape)
    # predict_data = predict_data[0:1024, :392, 21:373]
    # predict_data = predict_data[:, :, :608]
    predict_data = predict_data[xline_index, :, :]
    predict_data = np.moveaxis(predict_data, 0, -1)
    print('predict min', np.min(predict_data), np.max(predict_data))
    alpha1 = np.ones(shape=label_data.shape, dtype='float32')
    alpha2 = np.ones(shape=predict_data.shape, dtype='float32')
    alpha1[np.where(label_data < 0)] = 0
    alpha2[np.where(predict_data < 1)] = 0
    blockedge = horizon_line(predict_data)
    # alpha2[np.where(blockedge < 1)] = 0
    print("seismic_data shape:", seismic_data.shape)
    print("predict_data shape:", predict_data.shape)
    plt.subplot(1, 2, 1)
    plt.title('Seismic', fontsize=18)
    plt.imshow(seismic_data, cmap=plt.cm.gray)
    plt.imshow(label_data, cmap=plt.cm.rainbow, alpha=alpha1)
    # plt.imshow(label_data, cmap=plt.cm.rainbow)
    plt.subplot(1, 2, 2)
    plt.title('Predict', fontsize=18)
    # plt.imshow(seismic_data, cmap=plt.cm.gray)
    plt.imshow(predict_data, cmap=plt.cm.rainbow)
    # plt.imshow(label_data, cmap=plt.cm.ocean, alpha=alpha1)

    # plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + '0ilineblock.png', predict_data)
    # plt.imshow(blockedge)
    plt.show()
    return
def block_edge(predict_result):
    output = np.zeros(shape=(predict_result.shape[0], predict_result.shape[1], 4), dtype='float32')
    r = [0, 0, 0, 1, 1, 1]
    g = [0, 1, 1, 0, 0, 1]
    b = [1, 0, 1, 0, 1, 0]
    for i in range(predict_result.shape[0]-1):
        for j in range(predict_result.shape[1]):
            if predict_result[i, j] == predict_result[i+1, j]-1:
                output[i, j, 0] = r[int(predict_result[i, j])]
                output[i, j, 1] = g[int(predict_result[i, j])]
                output[i, j, 2] = b[int(predict_result[i, j])]
                output[i, j, 3] = 1
    return output
def horizon_line(predict_result):
    print('predict_result shape', predict_result.shape)
    output = np.zeros(shape=(predict_result.shape[0], predict_result.shape[1], 4), dtype='float32')
    output1 = np.zeros(shape=predict_result.shape, dtype='float32')
    output2 = np.zeros(shape=predict_result.shape, dtype='float32')
    output3 = np.zeros(shape=predict_result.shape, dtype='float32')
    output4 = np.zeros(shape=predict_result.shape, dtype='float32')
    r = [0, 0, 0, 1, 1, 1, 1]
    g = [0, 1, 1, 0, 0, 1, 1]
    b = [1, 0, 1, 0, 1, 0, 1]
    for i in range(7):
        a = np.where(predict_result == i)
        # print('a:', a, a[1].shape, np.max(a[0]), np.max(a[1]))
        output1[np.where(predict_result == i)] = r[i]
        output2[np.where(predict_result == i)] = g[i]
        output3[np.where(predict_result == i)] = b[i]
        output4[np.where(predict_result == i)] = 1
    output[:, :, 0] = output1
    output[:, :, 1] = output2
    output[:, :, 2] = output3
    output[:, :, 3] = output4
    return output
def miou(predict_file, label_file):
    predict_cube = np.load(predict_file)
    label_cube = np.load(label_file)
    predict_cube_2 = np.zeros(shape=predict_cube.shape, dtype='float32')
    mat = np.zeros((4, 4), dtype='int')
    for i in range(2):
        print(i)
        for j in range(4):
            if i == 0:
                a = predict_cube[j*100+105, :, :]
                for k in range(512):
                    for l in range(1,608):
                        if a[k, l]==1 and a[k, l-1] == 0:
                            predict_cube_2[j*100+105, k, l] = 1
                        elif a[k, l] == 2 and a[k, l-1] == 1:
                            predict_cube_2[j*100+105, k, l] = 2
                        elif a[k, l] == 3 and a[k, l-1] == 2:
                            predict_cube_2[j*100+105, k, l] = 3
            elif i == 2:
                a = predict_cube[:, j * 100 + 105, :]
                for k in range(512):
                    for l in range(1,608):
                        if a[k, l]==1 and a[k, l-1] == 0:
                            predict_cube_2[k, j*100+105, l] = 1
                        elif a[k, l] == 2 and a[k, l-1] == 1:
                            predict_cube_2[k, j*100+105, l] = 2
                        elif a[k, l] == 3 and a[k, l-1] == 2:
                            predict_cube_2[k, j*100+105, l] = 3
    for i in range(2):
        print(i)
        for j in range(4):
            if i == 0:
                a = predict_cube_2[j*100+105, :, :]
                b = label_cube[j*100+105, :, :]
                for k in range(512):
                    for l in range(608):
                        x1 = a[k, l]
                        x2 = b[k, l]
                        mat[int(x1), int(x2)] += 1
            elif i == 1:
                a = predict_cube_2[:, j * 100 + 105, :]
                b = label_cube[:, j * 100 + 105, :]
                for k in range(512):
                    for l in range(608):
                        x1 = a[k, l]
                        x2 = b[k, l]
                        mat[int(x1), int(x2)] += 1
    print('matrix:', mat)
    # plt.imshow(np.moveaxis(predict_cube_2[105, :, :], 0, -1))
    # plt.show()
    classes = ['0', '1', '2', '3']
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Oranges, vmax=2000)
    plt.title('confusion matrix')
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = mat.max()/2
    for i in range(4):
        for j in range(4):
            plt.text(j, i, format(mat[i, j]))
    plt.ylabel('Prediction')
    plt.xlabel("Ground Truth")
    plt.colorbar()
    plt.show()
    return
def inline_img_augmentation(predict_file, index, mode):
    data_cube = np.load(predict_file)
    # data_cube = np.moveaxis(data_cube, 0, -1)
    data_cube = data_cube[:1024, :384, 21:373]
    img = data_cube[320:704, index, :]
    img2 = np.zeros(shape=img.shape, dtype='float32')
    if mode == 0:
        output_img = img
    elif mode == 1:
        output_img = img[:, ::-1]
    elif mode == 2:
        output_img = img[::-1, :]
    elif mode == 3:
        output_img = img[::-1, ::-1]
    elif mode == 4:
        img2[:192, :] = img[:192, :]
        a = img[:192, :]
        img2[192:, :] = a[::-1, :]
        output_img = img2
    elif mode == 5:
        img2[192:, :] = img[:192, :]
        a = img[:192, :]
        img2[:192, :] = a[::-1, :]
        output_img = img2
    elif mode == 6:
        img2[192:, :] = img[192:, :]
        a = img[192:, :]
        img2[:192, :] = a[::-1, :]
        output_img = img2
    elif mode == 7:
        img2[:192, :] = img[192:, :]
        a = img[192:, :]
        img2[192:, :] = a[::-1, :]
        output_img = img2
    output_img = np.moveaxis(output_img, 0, -1)
    plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + str(index) + '_' + str(mode) + '.png', output_img)
    return
def predict_alpha_img(seismic_file, predict_file, index, mode):
    data_cube = np.load(seismic_file)
    data_cube = np.moveaxis(data_cube, 0, -1)
    data_cube = data_cube[:1024, :384, 21:373]
    predict_cube = np.load(predict_file)
    # seismic_img = np.moveaxis(data_cube[:, index, :], 0, -1)
    if mode == 'inline':
        predict_img = np.moveaxis(predict_cube[:, index, :], 0, -1)
    elif mode == 'xline':
        predict_img = np.moveaxis(predict_cube[index, :, :], 0, -1)
    # plt.imshow(seismic_img, cmap=plt.cm.gray)
    # plt.imshow(predict_img, alpha=0.5)
    # plt.show()
    plt.imsave(r'/home/limuyang/New_zealand_data/bishe_newZEALAND/' + str(index) + '_seismic.png', predict_img, cmap=plt.cm.gray)
    # a = []
    # b = []
    # c = data_cube[142, :, :]
    # for i in range(384):
    #     for j in range(352):
    #         a.append(c[i, j])
    # d = (c - np.mean(c)) / (np.var(c) ** 0.5)
    # for i in range(384):
    #     for j in range(352):
    #         b.append(d[i, j])
    # np.savetxt(r'a.txt', a)
    # np.savetxt(r'b.txt', b)
    return
def block_to_horizon(predict_file, index, mode):
    """
    将地层序列预测结果转化为目的层位预测结果
    :param predict_file: 地层序列预测结果
    :param index: 待显示剖面索引值，整型
    :param mode: 待显示剖面类型，'inline'或'xline'
    :return:
    """
    predict_cube = np.load(predict_file)
    if mode == 'inline':
        output_img = np.ones((1024, 352), dtype='float32')*9
        predict_img = predict_cube[:, index, :]
        for i in range(1024):
            for j in range(351):
                if predict_img[i, j] == predict_img[i, j+1]-1:
                    output_img[i, j+1] = predict_img[i, j]
        predict_img = np.moveaxis(predict_img, 0, -1)
        output_img = np.moveaxis(output_img, 0, -1)
    elif mode == 'xline':
        output_img = np.ones((384, 352), dtype='float32')*9
        predict_img = predict_cube[index, :, :]
        for i in range(384):
            for j in range(351):
                if predict_img[i, j] == predict_img[i, j+1]-1:
                    output_img[i, j+1] = predict_img[i, j]
        predict_img = np.moveaxis(predict_img, 0, -1)
        output_img = np.moveaxis(output_img, 0, -1)
    # plt.imsave(r'/home/limuyang/New_zealand_data/bishe_newZEALAND/' + str(index+1) + '_horizon2.png', output_img)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]#自定义色标范围，设置的色标为块状
    color_bar = ["purple", "blue", "cyan", "green", "yellow", "red"]#自定义色标颜色
    cmap = colors.ListedColormap(color_bar)#利用自定义颜色列表创建色标
    norms = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(output_img, cmap=cmap, norm=norms)#利用自定义色标显示
    cbar1 = plt.colorbar()
    cbar1.set_ticks([0, 1, 2, 3, 4, 5])#设置色标显示数值的位置
    cbar1.set_ticklabels(['sequence 0', 'sequence 1', 'sequence 2', 'sequence 3', 'sequence 4', 'sequence 5'])#设置色标显示的值
    # plt.imshow(predict_img, cmap=plt.cm.rainbow)
    plt.show()
    # plt.imsave(r'/home/limuyang/New_zealand_data/bishe_newZEALAND/' + str(index+1) + '_predict_2.png', predict_img)
    return
def label_trace_img(label_file, index, mode):
    """
    截图用函数
    :param label_file: 要成图的标签文件路径
    :param index: 要成图的剖面索引值，整型
    :param mode: 要成图的剖面类型，'inline'或'xline'
    :return:
    """
    label_cube = np.load(label_file)
    label_cube = label_cube[:1024, :384, 21:373]
    if mode == 'inline':
        output_img = np.ones((1024, 352), dtype='float32')*10
        label_img = label_cube[:, index, :]
        output_img[142, :] = label_img[142, :]
        output_img[342, :] = label_img[342, :]
        output_img[542, :] = label_img[542, :]
        output_img[742, :] = label_img[742, :]
        output_img[942, :] = label_img[942, :]
        output_img = np.moveaxis(output_img, 0, -1)
    elif mode == 'xline':
        output_img = np.ones((384, 352), dtype='float32')*10
        label_img = label_cube[index, :, :]
        output_img[46, :] = label_img[46, :]
        output_img[196, :] = label_img[196, :]
        output_img[346, :] = label_img[346, :]
        output_img = np.moveaxis(output_img, 0, -1)
    plt.imsave(r'/home/limuyang/New_zealand_data/bishe_newZEALAND/' + str(index) + '_trace.png', output_img)
    return
if __name__=='__main__':
    print('hello!')
    # check_inline(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"label_2.npy", r"predictAE.npy", 100)
    # label_img_edit(['0_iline.png', '1_iline.png', '2_iline.png'], 'label.npy')
    # seismic_cube_cut(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
    # check_inline(r"seismic_2.npy", r"label_2.npy", r"predictAE.npy", 46)
    # line_horizon_trans_block("label.npy")
    # check_inline_block(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"label_4.npy", r"label_4.npy", 46)
    # check_inline_block(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"/home/limuyang/New_zealand_data/label_3.npy", r"predictFreeze4_38.npy", 271)
    # check_xline_block(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"/home/limuyang/New_zealand_data/label_3.npy", r"predictFreeze4_38.npy", 642)
    # check_inline_block(r"grid_Amp.npy", r"cls_model.npy", r"predictFreeze.npy", 305)
    # block_add_fault("label_2.npy")
    # check_inline_block(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"label_4.npy", r"Ins_phase.npy", 46)
    # a = np.load(r"seis.npy")
    # print('a.shape:', a.shape)
    # miou(r"predictFreeze.npy", r"cls_model.npy")
    # inline_img_augmentation(r"/home/limuyang/New_zealand_data/label_3.npy", 46, 6)
    # predict_alpha_img(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"predictFreeze8_20.npy", 121, 'inline')
    block_to_horizon(r"/home/limuyang/New_zealand_data/label_3.npy", 46, 'inline')
    # label_trace_img(r"/home/limuyang/New_zealand_data/label_3.npy", 842, 'xline')
