import glob
import os
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

rootpath = 'dataset'
clinic_csv = os.path.join(rootpath, 'DLRF_v1.93.xlsx')


def make_datapath_list(rootpath):
    """데이터셋을 가지고와서 정면 사진과 측면 사진으로 분류하는 함수"""

    surgery_folder_path = os.path.join(rootpath, '1pre')
    no_surgery_folder_path = os.path.join(rootpath, '0pre')

    ''' 0. 전체 파일 리스트를 확보한다'''
    surgery = glob.glob(os.path.join(surgery_folder_path, '*.jpg'))
    no_surgery = glob.glob(os.path.join(no_surgery_folder_path, '*.jpg'))

    ''' 1. 정면과 측면 사진을 분류한다'''
    ### 정면
    preap_surgery = [f for f in surgery if 'preap' in f]  # 수술이 필요한 데이터셋
    preap_no_surgery = [f for f in no_surgery if 'preap' in f]  # 수술이 필요하지 않은 데이터셋

    ### 측면
    prelat_surgery = [f for f in surgery if 'prelat' in f]
    prelat_no_surgery = [f for f in no_surgery if 'prelat' in f]

    return preap_surgery, preap_no_surgery, prelat_surgery, prelat_no_surgery


def split_dataset(surgery, no_surgery, split_ratio=0.8):
    """데이터셋을 가지고와서 훈련과 테스트셋으로 분류하는 함수"""

    surgery_split_index = int(len(surgery) * split_ratio)
    no_surgery_split_index = int(len(no_surgery) * split_ratio)

    ### 정답 데이터셋을 train과 test로 분리한다.
    surgery_train = surgery[:surgery_split_index]
    surgery_test = surgery[surgery_split_index:]

    ### 오답 데이터셋을 train과 test로 분리한다.
    no_surgery_train = no_surgery[:no_surgery_split_index]
    no_surgery_test = no_surgery[no_surgery_split_index:]

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

    def __init__(self, resize, mean, std):  # , mean=None, std=None
        self.resize = resize
        self.mean = (0.485, 0.456, 0.406)  # mean
        self.std = (0.229, 0.224, 0.225)
        self.data_transform = {
            'train': A.Compose([
                A.Resize(height=resize, width=resize),
                A.RandomResizedCrop(height=resize, width=resize, scale=(0.5, 1.0)),  # 데이터 확장 1  # 데이터 확장 1
                A.HorizontalFlip(p=0.5),  # 데이터 확장 2
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # 어그멘테이션추가1
                # A.RandomGrayscale(p=0.1),#어그멘테이션추가2
                A.RandomBrightnessContrast(p=0.2),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),  # 추가추가
                # A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.5),#추가추가
                A.Solarize(threshold=128, p=0.5),  # 추가추가
                A.Normalize(mean=mean, std=std),  # 색상정보의 표준화,std mmean 뺄 떄 여기 부분
                ToTensorV2(),  # 텐서로의 변환
            ]),
            'val': A.Compose([
                A.Resize(height=resize, width=resize),  # 이미지 크기 조정
                A.Normalize(mean=mean, std=std),  # 색상 정보 표준화
                ToTensorV2()  # 넘파이 배열을 텐서로 변환
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        -----------
        phase : 'train' or 'val' 전처리 모드 지정
        """
        img = np.array(img)
        return self.data_transform[phase](image=img)['image']


class MedicalDataset(data.Dataset):

    def __init__(self, preap_img_list, prelat_img_list, phase, transform, clinic_csv):
        self.preap_img_list = preap_img_list
        self.prelat_img_list = prelat_img_list
        self.phase = phase
        self.transform = transform
        self.clinic_info = pd.read_excel(clinic_csv
                                         , usecols=['ID',
                                                    'Age\n(진료일기준)',
                                                    'BMI',
                                                    'Acceptability',
                                                    'Presence of Subsequent \nor concomittent fracture',
                                                    'AO OTA Classification']
                                         , header=2)

        # 정규화 하기 전에 원본 값을 저장
        self.clinic_info['Age_original'] = self.clinic_info['Age\n(진료일기준)']  # 원본 값 저장
        self.clinic_info['BMI_original'] = self.clinic_info['BMI']  # 원본 값 저장

        # 1. 나이를 정규화한다
        column = self.clinic_info['Age\n(진료일기준)']
        # scaler = StandardScaler() # Z-score Scaler 사용
        self.age_scaler = MinMaxScaler()  # Min-Max Scaler 사용
        self.clinic_info['Age\n(진료일기준)'] = self.age_scaler.fit_transform(column.values.reshape(-1, 1))

        # 2. AO OTA Classification를 0~1 사이의 값으로 수치화한다.
        label_encoder = LabelEncoder()
        scaler = MinMaxScaler()
        self.clinic_info['AO OTA Classification'] = label_encoder.fit_transform(
            self.clinic_info['AO OTA Classification'])  # 텍스트 컬럼을 정수로 변환
        self.clinic_info['AO OTA Classification'] = scaler.fit_transform(
            self.clinic_info['AO OTA Classification'].values.reshape(-1, 1))  # 텍스트 컬럼을 정수로 변환

        # 3. BMI 정규화.
        bmi_info = self.clinic_info['BMI']
        self.bmi_scaler = MinMaxScaler()
        self.clinic_info['BMI'] = self.bmi_scaler.fit_transform(bmi_info.values.reshape(-1, 1))
        # print("Clinic Info DataFrame Head: \n", self.clinic_info.head())  # 처음 몇 개의 행을 출력
        # print("Clinic Info Columns: \n", self.clinic_info.columns)  # 컬럼 이름 출력

        # # 데이터 확인
        # print(self.clinic_info.head())

    def get_scaler(self):
        return self.age_scaler, self.bmi_scaler

    def __len__(self):
        """이미지 개수를 반환"""
        return len(self.preap_img_list)

    def __getitem__(self, index):
        """전처리한 이미지의 텐서 형식의 데이터와 라벨 취득 """

        # 1. index번째 이미지 로드
        preap_file_path = self.preap_img_list[index]
        prelat_file_path = self.prelat_img_list[index]
        preap_img = Image.open(preap_file_path)  # [높이][넓이][색 RGB]
        prelat_img = Image.open(prelat_file_path)  # [높이][넓이][색 RGB]
        ## 파일 명 Check
        # print("preap_file_path == ", preap_file_path)
        # print("prelat_file_path == ", prelat_file_path)

        # 2. 이미지 전처리
        preap_img_transformed = self.transform(preap_img, self.phase).float()
        prelat_img_transformed = self.transform(prelat_img, self.phase).float()

        # 3. Label 정보 추출
        label = 0
        if '1pre' in preap_file_path:
            label = 1

        # 4. 환자정보 추출 (Text)
        filename = preap_file_path.split('\\')[2]
        id = 'DLRF-' + filename.split('_')[1]
        clinic_info_byID = self.clinic_info[self.clinic_info['ID'] == id]
        # print(clinic_info_byID.iloc[0].values)
        age = clinic_info_byID['Age\n(진료일기준)'].iloc[0]
        # if not clinic_info_byID.empty and 'Age\n(진료일기준)' in clinic_info_byID.columns:
        #     age = clinic_info_byID['Age\n(진료일기준)'].iloc[0]
        # else:
        #     print("데이터가 없거나 열 이름이 잘못되었습니다.")
        bmi = clinic_info_byID['BMI'].iloc[0]
        acceptability = clinic_info_byID['Acceptability'].iloc[0]
        subsequent = clinic_info_byID['Presence of Subsequent \nor concomittent fracture'].iloc[0]
        ota = clinic_info_byID['AO OTA Classification'].iloc[0]

        clinic_info = [age, bmi]
        clinic_info = torch.tensor(clinic_info, dtype=torch.float32)

        return id, preap_img_transformed, prelat_img_transformed, clinic_info, label


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
    print('trainset_preap_num', len(train_list_preap))
    train_list_prelat = prelat_surgery_train + prelat_no_surgery_train
    print('trainset_prelat_num', len(train_list_prelat))
    val_list_preap = preap_surgery_val + preap_no_surgery_val
    print('valset_preap_num', len(val_list_preap))
    val_list_prelat = prelat_surgery_val + prelat_no_surgery_val
    print('valset_prelat_num', len(val_list_prelat))
    test_list_preap = preap_surgery_test + preap_no_surgery_test
    print('testset_preap_num', len(test_list_preap))
    test_list_prelat = prelat_surgery_test + prelat_no_surgery_test
    print('testset_prelat_num', len(test_list_prelat))
    train_dataset = MedicalDataset(preap_img_list=train_list_preap,
                                   prelat_img_list=train_list_prelat,
                                   transform=ImageTransform(resize, mean, std),
                                   phase='train',
                                   clinic_csv=clinic_csv)

    val_dataset = MedicalDataset(preap_img_list=val_list_preap,
                                 prelat_img_list=val_list_prelat,
                                 transform=ImageTransform(resize, mean, std),
                                 phase='val',
                                 clinic_csv=clinic_csv)

    test_dataset = MedicalDataset(preap_img_list=test_list_preap,
                                  prelat_img_list=test_list_prelat,
                                  transform=ImageTransform(resize, mean, std),
                                  phase='val',
                                  clinic_csv=clinic_csv)

    # 1. 정면 사진의 데이터로더 구성
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 2. 사전형 변수에 정리
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    return dataloaders_dict, test_dataloader

# if __name__ == "__main__":
#     # 테스트를 위해 DataLoader 생성
#     resize = 224
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     batch_size = 4

#     dataloaders_dict, test_dataloader = get_dataloader(resize, mean, std, batch_size)

#     # 데이터셋이 잘 로드되었는지 확인하기 위해 한 배치만 출력
#     for preap_inputs, prelat_inputs, clinic_inputs, labels in test_dataloader:
#         print(f"PreAP Inputs Shape: {preap_inputs.shape}")
#         print(f"PreLat Inputs Shape: {prelat_inputs.shape}")
#         print(f"Clinic Info: {clinic_inputs}")
#         print(f"Labels: {labels}")
#         break  # 한 배치만 확인

