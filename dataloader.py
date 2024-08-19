import glob
import os
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


rootpath = 'dataset'

def make_datapath_list(rootpath):

    """데이터셋을 가지고와서 정면 사진과 측면 사진으로 분류하는 함수"""

    surgery_folder_path = os.path.join(rootpath, '1pre')
    no_surgery_folder_path = os.path.join(rootpath, '0pre')

    ''' 0. 전체 파일 리스트를 확보한다'''
    surgery = glob.glob(os.path.join(surgery_folder_path, '*.jpg'))
    no_surgery = glob.glob(os.path.join(no_surgery_folder_path, '*.jpg'))

    ''' 1. 정면과 측면 사진을 분류한다'''
    ### 정면
    preap_surgery = [f for f in surgery if 'preap' in f]        # 수술이 필요한 데이터셋
    preap_no_surgery = [f for f in no_surgery if 'preap' in f]  # 수술이 필요하지 않은 데이터셋

    ### 측면
    prelat_surgery = [f for f in surgery if 'prelat' in f]
    prelat_no_surgery = [f for f in no_surgery if 'prelat' in f]

    return preap_surgery, preap_no_surgery, prelat_surgery, prelat_no_surgery

def split_dataset(surgery, no_surgery, split_ratio=0.7):

    """데이터셋을 가지고와서 훈련과 테스트셋으로 분류하는 함수"""

    split_index = int(len(surgery) * split_ratio)

    ### 정답 데이터셋을 train과 test로 분리한다.
    surgery_train = surgery[:split_index]
    surgery_test = surgery[split_index:]

    ### 오답 데이터셋을 train과 test로 분리한다.
    no_surgery_train = no_surgery[:split_index]
    no_surgery_test = no_surgery[split_index:]

    return surgery_train, surgery_test, no_surgery_train, no_surgery_test



class ImageTransform():
    """
    이미지 전처리 클래스. train과 val동작이 다르다.
    train시에는 RandomResizedCrop과 RandomHorizontalFlip으로 데이터를 확장한다.
    이미지 크기를 리사이즈하고 색상을 표준화한다.

    Attributes
    -------------
    resize : int
        리사이즈 대상 이미지의 크기
    mean : (R, G, B)
        각 색상 채널의 평균값
    std : (R, G, B)
        각 색상 채널의 표준편차
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),  # 데이터 확장 1
                transforms.RandomHorizontalFlip(),  # 데이터 확장 2
                transforms.ToTensor(),  # 텐서로의 변환
                #transforms.Normalize(mean, std)  # 색상정보의 표준화
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # 리사이즈
                transforms.CenterCrop(resize),  # 이미지 중앙을 Resize x Resize로 자른다.
                transforms.ToTensor(),  # 텐서로의 변환
                # transforms.Normalize(mean, std)  # 색상정보의 표준화
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        -----------
        phase : 'train' or 'val' 전처리 모드 지정
        """
        return self.data_transform[phase](img)


class MedicalDataset(data.Dataset):

    def __init__(self, preap_img_list, prelat_img_list, phase, transform):
        self.preap_img_list = preap_img_list
        self.prelat_img_list = prelat_img_list

        self.phase = phase
        self.transform = transform

    def __len__(self):
        """이미지 개수를 반환"""
        return len(self.preap_img_list)

    def __getitem__(self, index):
        """전처리한 이미지의 텐서 형식의 데이터와 라벨 취득 """

        # 1. index번째 이미지 로드
        preap_file_path = self.preap_img_list[index]
        prelat_file_path = self.prelat_img_list[index]
        preap_img = Image.open(preap_file_path)   # [높이][넓이][색 RGB]
        prelat_img = Image.open(prelat_file_path)  # [높이][넓이][색 RGB]

        ## 파일 명 Check
        # print("preap_file_path == ", preap_file_path)
        # print("prelat_file_path == ", prelat_file_path)

        # 2. 이미지 전처리
        preap_img_transformed = self.transform(preap_img, self.phase)
        prelat_img_transformed = self.transform(prelat_img, self.phase)

        # 3. Label 정보 추출
        label = 0
        if '1pre' in preap_file_path:
            label = 1

        return preap_img_transformed, prelat_img_transformed, label

def get_dataloader(resize, mean, std, batch_size):

    preap_surgery, preap_no_surgery, prelat_surgery, prelat_no_surgery = make_datapath_list(rootpath=rootpath)

    # 1. 정면 데이터셋을 train과 test로 분리한다.
    preap_surgery_trainA, preap_surgery_test, preap_no_surgery_trainA, preap_no_surgery_test = split_dataset(
        preap_surgery, preap_no_surgery, split_ratio=0.8)

    # 2. 측면 데이터셋을 train과 test로 분리한다.
    prelat_surgery_trainA, prelat_surgery_test, prelat_no_surgery_trainA, prelat_no_surgery_test = split_dataset(
        prelat_surgery, prelat_no_surgery, split_ratio=0.8)

    # 3. 정면 train dataset 중에 validation dataset을 확보한다.
    preap_surgery_train, preap_surgery_val, preap_no_surgery_train, preap_no_surgery_val = split_dataset(
        preap_surgery_trainA, preap_no_surgery_trainA, split_ratio=0.8)

    # 4. 측면 train dataset 중에 validation dataset을 확보한다.
    prelat_surgery_train, prelat_surgery_val, prelat_no_surgery_train, prelat_no_surgery_val = split_dataset(
        prelat_surgery_trainA, prelat_no_surgery_trainA, split_ratio=0.8)

    # 5. 훈련데이터와 테스트 데이터를 불러온다.
    train_list_preap = preap_surgery_train + preap_no_surgery_train
    train_list_prelat = prelat_surgery_train + prelat_no_surgery_train

    val_list_preap = preap_surgery_val + preap_no_surgery_val
    val_list_prelat = prelat_surgery_val + prelat_no_surgery_val

    test_list_preap = preap_surgery_test + preap_no_surgery_test
    test_list_prelat = prelat_surgery_test + prelat_no_surgery_test

    # print("훈련데이터 개수 == ", len(train_list_preap + train_list_prelat))
    # print("검증데이터 개수 == ", len(val_list_preap + val_list_prelat))
    # print("테스트데이터 개수 ==", len(test_list_preap + test_list_prelat))


    train_dataset = MedicalDataset(preap_img_list=train_list_preap,
                                         prelat_img_list=train_list_prelat,
                                         transform=ImageTransform(resize, mean, std),
                                         phase='train')

    val_dataset = MedicalDataset(preap_img_list=val_list_preap,
                                   prelat_img_list=val_list_prelat,
                                   transform=ImageTransform(resize, mean, std),
                                   phase='val')

    test_dataset = MedicalDataset(preap_img_list=test_list_preap,
                                 prelat_img_list=test_list_prelat,
                                 transform=ImageTransform(resize, mean, std),
                                 phase='val')

    # 1. 정면 사진의 데이터로더 구성
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    # 2. 사전형 변수에 정리
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    return dataloaders_dict, test_dataloader
