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


#난수 시드 설정
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)




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
                transforms.Normalize(mean, std)  # 색상정보의 표준화
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # 리사이즈
                transforms.CenterCrop(resize),  # 이미지 중앙을 Resize x Resize로 자른다.
                transforms.ToTensor(),  # 텐서로의 변환
                transforms.Normalize(mean, std)  # 색상정보의 표준화
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

    def __init__(self, img_list, phase, transform):
        self.img_list = img_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        """이미지 개수를 반환"""
        return len(self.img_list)

    def __getitem__(self, index):
        """전처리한 이미지의 텐서 형식의 데이터와 라벨 취득 """

        # 1. index번째 이미지 로드
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [높이][넓이][색 RGB]

        # 2. 이미지 전처리
        img_transformed = self.transform(img, self.phase)  # torch.Size([3, 224, 224])

        # 3. Label 정보 추출
        label = 0
        if '1pre' in image_file_path:
            label = 1

        return img_transformed, label


