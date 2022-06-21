import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
def horizon_check(horizon):
    plt.imshow(horizon)
    plt.show()
    return
def fault_img(fault):
    for i in range(3):
        plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + str(i) + 'inline_fault.png', fault[1][:, :, i])
    for i in range(5):
        plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + str(i) + 'xline_fault.png', fault[0][:, i, :])
    return
def horizon_img(horizon_list, seismic_cube):
    # figure = plt.figure(dpi=300)
    # inline = np.zeros((501, 1043), dtype='float32')
    # xline = np.zeros((501, 396), dtype='float32')
    for i in range(3):
        inline = np.zeros((501, 1043), dtype='float32')
        alpha = np.ones((501, 1043), dtype='float32')
        for j in range(5):
            inline += horizon_list[j*2][:, :, i] * (j + 1)
            # plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + str(i)+ 'inline' + str(j) + 'horizon' + '.png', horizon_list[j*2][:, :, i])
        # plt.imshow(seismic_cube[21:, :, i * 150 + 46], cmap=plt.cm.gray)
        # inline[np.where(inline<=0)] = np.nan
        # alpha[np.where(inline<1)] = 0
        # plt.imshow(inline, alpha=alpha)
        # plt.axis('off')
        # plt.show()
        plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + str(i) + '_iline.png', inline)
        # plt.imsave(str(i) + '_iline_seismic.png', seismic_cube[:, :, i * 150 + 46], cmap=plt.cm.gray)
        # plt.savefig(r"/home/limuyang/New_zealand_data/Label_fig/inline" + str(i * 150 + 46) + '.png')
        # plt.cla()
        # plt.clf()
    for i in range(5):
        xline = np.zeros((501, 396), dtype='float32')
        alpha = np.ones((501, 396), dtype='float32')
        for j in range(5):
            # xline += horizon_list[j*2+1][:, i, :] * (j + 1)
            plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + str(i)+ 'xline' + str(j) + 'horizon' + '.png', horizon_list[j*2+1][:, i, :])
        # plt.imshow(seismic_cube[:, i * 200 + 142, :], cmap=plt.cm.gray)
        # xline[np.where(xline<=0)] = np.nan
        # alpha[np.where(xline<1)] = 0
        # plt.imshow(xline, alpha=alpha)
        # plt.show()
        # plt.imsave(str(i) + '_xline.png', xline)
        # plt.imsave(str(i) + '_xline_seismic.png', seismic_cube[:, i * 200 + 142, :], cmap=plt.cm.gray)
        # plt.savefig(r"/home/limuyang/New_zealand_data/Label_fig/xline" + str(i * 200 + 142) + '.png')
        # plt.cla()
        # plt.clf()
    return
def horizon_img2(horizon_list):
    horizon_img = []
    for i in range(3):
        inline = np.zeros(shape=(501, 1043), dtype='float32')
        for j in range(5):
            inline += horizon_list[j * 2][:, :, i] * (j + 1)
        horizon_img.append(inline)
    for i in range(5):
        xline = np.zeros(shape=(501, 396), dtype='float32')
        for j in range(5):
            xline += horizon_list[j*2+1][:, i, :] * (j + 1)
        horizon_img.append(xline)
    return horizon_img
def label_generate(horizon_list):
    horizon_img = horizon_img2(horizon_list=horizon_list)
    label_cube = np.zeros(shape=(501, 1043, 396), dtype='float32') - 1
    for i in range(3):
        label_cube[:, :, 150 * i + 46] = horizon_img[i]
    for i in range(5):
        label_cube[:, 200 * i + 142] = horizon_img[i+3]
    # plt.imshow(label_cube[:, 142, :])
    # plt.show()
    label_cube = np.moveaxis(label_cube, 0, -1)
    print(r"label_cube's shape", label_cube.shape)
    np.save(r"label.npy", label_cube)
    return
def r_list_append(r_list, append_list):
    output_list = r_list * 1
    for i in range(len(append_list)):
        output_list.append(append_list[i])
    shuffle(output_list)
    return output_list
if __name__ == "__main__":
    seismic = np.load(r"Seismic/Opunake_Quad_A.npy")
    print("Seismic's type and shape:", type(seismic), seismic.shape)
    horizon_list = []
    fault_list = []
    horizon_file_name = ["Horizon_1_3Inline.npy", "Horizon_1_5Xline.npy", "Horizon_2_3Inline.npy",
                         "Horizon_2_5Xline.npy", "Horizon_3_3Inline.npy", "Horizon_3_5Xline.npy",
                         "Horizon_4_3Inline.npy", "Horizon_4_5Xline.npy", "Horizon_5_3Inline.npy", "Horizon_5_5Xline.npy"]
    fault_file_name = ["Fault_5Xline.npy", "Fault_3Inline.npy"]
    for i in range(10):
        new_horizon = np.load(r"Labels/horizon/" + horizon_file_name[i])
        print(horizon_file_name[i][:-4] + r"'s shape:", new_horizon.shape)
        horizon_list.append(new_horizon)
    for i in range(2):
        new_fault = np.load(r"Labels/fault/" + fault_file_name[i])
        print(fault_file_name[i][:-4] + r"'s shape:", new_fault.shape)
        fault_list.append(new_fault)
    # horizon_check(horizon_list[0][:, :, 0])
    # horizon_img(horizon_list, seismic)
    fault_img(fault_list)
    # label_generate(horizon_list=horizon_list)
    # print('horizon_list[0].shape:', horizon_list[0].shape)



