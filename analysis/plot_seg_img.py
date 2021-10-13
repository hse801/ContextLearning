import numpy as np
import SimpleITK as sitk
import os
import scipy
import scipy.ndimage as ndimage
import torchio as tio
import glob
from torch.nn import functional as F
from PIL import Image
from typing import List, Tuple
from PIL import Image
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2

lung_path = glob.glob('F:/LungCancerData/valid/27246762/')


def idx2rgb(arr: np.array, palette: List[Tuple[int, int, int]]):
    # n = len(palette)
    # arr.size = h x w
    # 0 <= arr[y][x] < n
    h, w = arr.shape
    print(f'h = {h}, w = {w}')
    # rgb = np.zeros([h, w, 3], dtype=np.uint8)
    rgb = np.zeros([h, w, 3])
    for i in range(int(h)):
        for j in range(int(w)):
            # print(f'type i = {type(i)}, type j = {type(j)}')
            rgb[i][j] = palette[int(arr[i][j])]
    return rgb

# def create_pred_img()
for path in lung_path:
    pred_file = glob.glob(path + 'Co_Con_*')
    # pred_file = glob.glob(path + 'ConResNet_Pred*')
    # pred_file = glob.glob(path + 'pred_RESUNET*')
    os.chdir(path)

    mask_path = path + 'ROI_cut.nii.gz'
    lymph_path = path + 'lymph_cut_sum.nii.gz'
    # mask_file = path + 'crop_mask.nii.gz'
    # print(f'mask file = {mask_path}')

    if os.path.isfile(lymph_path) and os.path.isfile(mask_path):
        # ROI and lymph
        mask_img = sitk.ReadImage(mask_path)
        img_mask_data = sitk.GetArrayFromImage(mask_img)
        img_lymph = sitk.ReadImage(lymph_path)
        img_lymph_data = sitk.GetArrayFromImage(img_lymph)

        img_mask_data = img_mask_data + img_lymph_data
        img_mask_data[img_mask_data > 0] = 1
    elif os.path.isfile(mask_path):
        # only ROI
        mask_img = sitk.ReadImage(mask_path)
        img_mask_data = sitk.GetArrayFromImage(mask_img)
        img_mask_data = img_mask_data
        img_mask_data[img_mask_data > 0] = 1
    else:
        # only lymph
        img_lymph = sitk.ReadImage(lymph_path)
        img_lymph_data = sitk.GetArrayFromImage(img_lymph)
        img_mask_data = img_lymph_data
        img_mask_data[img_mask_data > 0] = 1

    img_pred = sitk.ReadImage(pred_file[0])
    arr_pred_data = sitk.GetArrayFromImage(img_pred)

    os.chdir(path)
    TP = 1
    TN = 2
    FP = 3
    FN = 4
    slice_num = 50

    color_map = {
        'TP': (255/255,228/255,196/255),
        'TN': (255/255,255/255,255/255),
        'FP': (255/255,69/255,0/255),
        'FN': (0/255,0/255,205/255)
    }

    # color_map = {
    #     'TP': (175/255,238/255,238/255),
    #     'TN': (255/255,255/255,255/255),
    #     'FP': (255/255,69/255,0/255),
    #     'FN': (0/255,128/255,0/255)
    # }

    # color_map = {
    #     'TP': (255/255,228/255,196/255),
    #     'TN': (255/255,255/255,255/255),
    #     'FP': (255/255,69/255,0/255),
    #     'FN': (0/255,0/255,205/255)
    # }

    # ResUNet
    compare_pred = np.zeros((80, 128, 160))
    compare_pred = np.where((img_mask_data == 1) & (arr_pred_data == 1), TP, compare_pred)  # True Positive
    compare_pred = np.where((img_mask_data == 0) & (arr_pred_data == 0), TN, compare_pred)  # True Negative
    compare_pred = np.where((img_mask_data == 0) & (arr_pred_data == 1), FP, compare_pred)  # False Positive
    compare_pred = np.where((img_mask_data == 1) & (arr_pred_data == 0), FN, compare_pred)  # False Negative

    pred_slice = compare_pred[slice_num, :, :]
    map_pred = np.ones((128, 160, 3))

    map_pred[pred_slice == 1] = color_map['TP']
    map_pred[pred_slice == 2] = color_map['TN']
    map_pred[pred_slice == 3] = color_map['FP']
    map_pred[pred_slice == 4] = color_map['FN']
    plt.axis("off")

    plt.gca().invert_yaxis()
    plt.imshow(map_pred)
    plt.savefig(r"Co_Con.jpg", bbox_inches='tight', pad_inches=0)
    plt.show()

    print(f'image saved in {os.getcwd()}')

