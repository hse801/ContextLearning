import glob
import os
import csv

"""
'E:/HSE/lymphdata/*/'
'E:/HSE/LungCancerData/*/*/'

lymphdata 에만 있는 파일 목록 찾기

"""

path_list = ('C:/Users/Bohye/data/', 'D:/03. DataSet/Lung Cancer (PET-CT)/SNUH_lung')
file_list = []
overlap_list = []

for p in path_list:
    for f in os.listdir(p):
    # file = os.listdir(p)
        if f not in file_list:
            file_list.append(f)
        else:
            overlap_list.append(f)
    # print(file)

print(f'type = {type(file_list)}, {len(file_list)}')
file_set = set(file_list)
print(f'list num = {len(file_list)}, set num = {len(file_set)}')
print(f'overlap num = {len(overlap_list)}, {overlap_list}')

folder_list = glob.glob('E:/HSE/LungCancerData/*/*/')
lymph_path = glob.glob('E:/HSE/lymphdata/*/')
print(f'len = {len(folder_list)}, list = {folder_list}')
name_list = []
for f in folder_list:
    name_list.append(f.split('\\')[-2])
print(f'len = {len(name_list)}, list = {name_list}')
absent_list = []
f = open('excluded_files.csv', 'w')
wr = csv.writer(f)
with open('excluded_files.csv', 'w') as f:

    for l in lymph_path:
        lymph_name = l.split('\\')[-2]
        if lymph_name not in name_list:
            absent_list.append(lymph_name)
    wr.writerow(absent_list)

print(f'len = {len(absent_list)}, list = {absent_list}')

# f.close()
