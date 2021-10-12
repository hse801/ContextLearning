import numpy as np
import SimpleITK as sitk
import os
import scipy
import scipy.ndimage as ndimage
import torchio as tio
import glob
from torch.nn import functional as F

lung_path = glob.glob('F:/LungCancerData/valid/27246762/')

for path in lung_path:
    CoCon_file = glob.glob(path + 'Co_Con_*')
    ConRes_file = glob.glob(path + 'ConResNet_Pred*')
    ResU_file = glob.glob(path  + 'pred_RESUNET*')
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

    img_CoCon = sitk.ReadImage(CoCon_file[0])
    arr_CoCon_data = sitk.GetArrayFromImage(img_CoCon)

    img_ConRes = sitk.ReadImage(ConRes_file[0])
    arr_ConRes_data = sitk.GetArrayFromImage(img_ConRes)

    img_ResU = sitk.ReadImage(ResU_file[0])
    arr_ResU_data = sitk.GetArrayFromImage(img_ResU)

    os.chdir(path)
    diff_CoCon = img_mask_data - arr_CoCon_data
    diff_CoCon = np.where(diff_CoCon != 0, 1, 0)
    diff_CoCon_img = sitk.GetImageFromArray(diff_CoCon)
    sitk.WriteImage(diff_CoCon_img[:, :, :], 'diff_CoConAspp.nii.gz')

    diff_ConRes = img_mask_data - arr_ConRes_data
    diff_ConRes = np.where(diff_ConRes != 0, 1, 0)
    diff_ConRes_img = sitk.GetImageFromArray(diff_ConRes)
    sitk.WriteImage(diff_ConRes_img[:, :, :], 'diff_ConRes.nii.gz')

    diff_ResU = img_mask_data - arr_ResU_data
    diff_ResU = np.where(diff_ResU != 0, 1, 0)
    diff_ResU_img = sitk.GetImageFromArray(diff_ResU)
    sitk.WriteImage(diff_ResU_img[:, :, :], 'diff_ResU.nii.gz')


    print(f'Difference file saved in {os.getcwd()}')

    # break
