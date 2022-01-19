import SimpleITK as sitk
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms, utils
# import lib.augment3D as augment3D
# import torchio as tio
from skimage.transform import resize
import os
import matplotlib.pyplot as plt


def sum_pet_ct(ct_file, pet_file, ct_ratio):

    img_ct = sitk.ReadImage(ct_file)
    img_ct_data = sitk.GetArrayFromImage(img_ct)
    img_ct_data = (img_ct_data - np.mean(img_ct_data)) / (np.std(img_ct_data) + 1e-8)

    img_pet = sitk.ReadImage(pet_file)
    img_pet_data = sitk.GetArrayFromImage(img_pet)
    img_pet_data = (img_pet_data - np.mean(img_pet_data)) / (np.std(img_pet_data) + 1e-8)

    pet_ct_sum = ct_ratio * img_ct_data + (1 - ct_ratio) * img_pet_data
    pet_ct_sum = (pet_ct_sum - np.mean(pet_ct_sum)) / (np.std(pet_ct_sum) + 1e-8)

    sum_img = sitk.GetImageFromArray(pet_ct_sum)
    sitk.WriteImage(sum_img, 'pet_ct_sum.nii.gz')

    # fig = plt.figure(figsize=(24, 24))
    # fig.subplots_adjust(left=0, right=1, top=0.8)
    # for i in range(80):
    #     fig.add_subplot(8, 10, i + 1, xticks=[], yticks=[])
    #     plt.imshow(pet_ct_sum[i, :, :], cmap='jet')
    # plt.savefig('F:/ContextLearning/figures/pet_ct_sum.jpg', bbox_inches='tight')


test_folder_path = glob.glob('F:/LungCancerData/test/*/')

for f in test_folder_path:
    ct_path = glob.glob(f + 'CT_cut.nii.gz')
    pet_path = glob.glob(f + 'PET_cut.nii.gz')
    print(f'ct path = {ct_path}')
    sum_pet_ct(ct_file=ct_path[0], pet_file=pet_path[0], ct_ratio=0.5)
    break