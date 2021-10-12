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
    ResU_file = glob.glob(path + 'pred_RESUNET*')
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
    TN = 5
    TP = 15
    FN = 7
    FP = -3

    '''
    plot difference for each model

    diff_CoCon = img_mask_data - arr_CoCon_data
    diff_CoCon = np.where((img_mask_data != 0) & (arr_CoCon_data == 0), FN, diff_CoCon) # False Negative
    diff_CoCon = np.where(img_mask_data < arr_CoCon_data, FP, diff_CoCon)  # False Positive
    diff_CoCon = np.where(diff_CoCon == 0, TN, diff_CoCon) # True Negative
    diff_CoCon = np.where(diff_CoCon == 0, TP, diff_CoCon) # True Positive
    diff_CoCon_img = sitk.GetImageFromArray(diff_CoCon)
    sitk.WriteImage(diff_CoCon_img[:, :, :], 'diff_TPFP_CoConAspp.nii.gz')

    diff_ConRes = img_mask_data - arr_ConRes_data
    diff_ConRes = np.where(diff_ConRes > 0, FN, diff_ConRes) # False Negative
    diff_ConRes = np.where(diff_ConRes == -1, FP, diff_ConRes)  # False Positive
    diff_ConRes = np.where(diff_ConRes == 0, TN, diff_ConRes)
    diff_ConRes_img = sitk.GetImageFromArray(diff_ConRes)
    sitk.WriteImage(diff_ConRes_img[:, :, :], 'diff_TPFP_ConRes.nii.gz')
    
    diff_ResU = img_mask_data - arr_ResU_data
    diff_ResU = np.where(diff_ResU > 0, FN, diff_ResU) # False Negative
    diff_ResU = np.where(diff_ResU == -1, FP, diff_ResU)  # False Positive
    diff_ResU = np.where(diff_ResU == 0, TN, diff_ResU)
    diff_ResU_img = sitk.GetImageFromArray(diff_ResU)
    sitk.WriteImage(diff_ResU_img[:, :, :], 'diff_TPFP_ResU.nii.gz')
    '''

    compare_ResU = np.zeros((80, 128, 160))
    diff_ResU = img_mask_data - arr_ResU_data
    # diff_ResU = np.where(diff_ResU != 0, 1, 0)
    compare_ResU = np.where((img_mask_data == 1) & (arr_ResU_data == 0), FN, compare_ResU) # False Negative
    compare_ResU = np.where((img_mask_data == 0) & (arr_ResU_data == 1), FP, compare_ResU)  # False Positive
    compare_ResU = np.where((img_mask_data == 0) & (arr_ResU_data == 0), TN, compare_ResU) # True Negative
    compare_ResU = np.where((img_mask_data == 1) & (arr_ResU_data == 1), TP, compare_ResU) # True Positive
    compare_ResU_img = sitk.GetImageFromArray(compare_ResU)
    sitk.WriteImage(compare_ResU_img[:, :, :], 'Compare_ResU.nii.gz')

    compare_CoCon = np.zeros((80, 128, 160))
    diff_CoCon = img_mask_data - arr_CoCon_data
    # diff_ResU = np.where(diff_ResU != 0, 1, 0)
    compare_CoCon = np.where((img_mask_data == 1) & (arr_CoCon_data == 0), FN, compare_CoCon) # False Negative
    compare_CoCon = np.where((img_mask_data == 0) & (arr_CoCon_data == 1), FP, compare_CoCon)  # False Positive
    compare_CoCon = np.where((img_mask_data == 0) & (arr_CoCon_data == 0), TN, compare_CoCon) # True Negative
    compare_CoCon = np.where((img_mask_data == 1) & (arr_CoCon_data == 1), TP, compare_CoCon) # True Positive
    compare_CoCon_img = sitk.GetImageFromArray(compare_CoCon)
    sitk.WriteImage(compare_CoCon_img[:, :, :], 'Compare_CoCon.nii.gz')

    compare_ConRes = np.zeros((80, 128, 160))
    diff_ConRes = img_mask_data - arr_ConRes_data
    # diff_ResU = np.where(diff_ResU != 0, 1, 0)
    compare_ConRes = np.where((img_mask_data == 1) & (arr_ConRes_data == 0), FN, compare_ConRes) # False Negative
    compare_ConRes = np.where((img_mask_data == 0) & (arr_ConRes_data == 1), FP, compare_ConRes)  # False Positive
    compare_ConRes = np.where((img_mask_data == 0) & (arr_ConRes_data == 0), TN, compare_ConRes) # True Negative
    compare_ConRes = np.where((img_mask_data == 1) & (arr_ConRes_data == 1), TP, compare_ConRes) # True Positive
    compare_ConRes_img = sitk.GetImageFromArray(compare_ConRes)
    sitk.WriteImage(compare_ConRes_img[:, :, :], 'Compare_ConRes.nii.gz')

    print(f'Difference file saved in {os.getcwd()}')

    # break
