import numpy as np
from dataloader import MedicalDataset,ImageTransform,make_datapath_list,split_dataset
from utils import imshow
from train import train_model
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 난수 시드 고정
    torch.manual_seed(42)


    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(resize, mean, std)


    preap_surgery, preap_no_surgery, prelat_surgery, prelat_no_surgery = make_datapath_list('dataset')

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

    # 1. 정면 사진의 훈련데이터셋 구성
    preap_train_dataset = MedicalDataset(img_list=train_list_preap, transform=ImageTransform(resize, mean, std),
                                         phase='train')

    # 1.1 정면 사진의 Val 데이터셋 구성
    preap_val_dataset = MedicalDataset(img_list=val_list_preap, transform=ImageTransform(resize, mean, std),
                                       phase='val')

    # 1.2 정면 사진의 Test 데이터셋 구성
    preap_test_dataset = MedicalDataset(img_list=test_list_preap, transform=ImageTransform(resize, mean, std),
                                        phase='val')

    # 2. 측면 사진의 훈련데이터셋 구성
    prelat_train_dataset = MedicalDataset(img_list=train_list_prelat, transform=ImageTransform(resize, mean, std),
                                          phase='train')

    # 2.2 측면 사진의 Val 데이터셋 구성
    prelat_val_dataset = MedicalDataset(img_list=val_list_prelat, transform=ImageTransform(resize, mean, std),
                                        phase='val')

    # 2.2 측면 사진의 Test 데이터셋 구성
    prelat_test_dataset = MedicalDataset(img_list=test_list_prelat, transform=ImageTransform(resize, mean, std),
                                         phase='val')

    batch_size = 5

    # 1. 정면 사진의 데이터로더 구성
    preap_train_dataloader = torch.utils.data.DataLoader(preap_train_dataset, batch_size=batch_size, shuffle=True)
    preap_val_dataloader = torch.utils.data.DataLoader(preap_val_dataset, batch_size=batch_size, shuffle=True)
    preap_test_dataloader = torch.utils.data.DataLoader(preap_test_dataset, batch_size=batch_size, shuffle=True)

    # 2. 측면 사진의 데이터로더 구성
    prelat_train_dataloader = torch.utils.data.DataLoader(prelat_train_dataset, batch_size=batch_size, shuffle=True)
    prelat_val_dataloader = torch.utils.data.DataLoader(prelat_train_dataset, batch_size=batch_size, shuffle=True)
    prelat_test_dataloader = torch.utils.data.DataLoader(prelat_test_dataset, batch_size=batch_size, shuffle=True)

    # 3. 사전형 변수에 정리
    preap_dataloaders_dict = {"train": preap_train_dataloader, "val": preap_val_dataloader}
    prelat_dataloaders_dict = {"train": prelat_train_dataloader, "val": prelat_val_dataloader}

    #  1. 가벼운 테스트를 위해 우선 pre-trained model을 사용한다
    net = models.resnet50(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)  # num_classes는 새로운 데이터셋의 클래스 수
    net.to(device)

    # 훈련 모드로 설정
    net.train()

    for name, param in net.named_parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    # 최적화 기법 설정
    # optimizer = optim.Adam(net.fc.parameters(), lr=0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 100

    # 학습 및 검증 실시
    net = train_model(net, preap_dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in preap_test_dataloader:
            images, labels = data

            net.eval()

            # GPU가 사용가능하면 GPU에 데이터 전송
            net.to(device)
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    save_path = './weights_transfer_learning.pth'
    torch.save(net.state_dict(), save_path)  # 저장할 파라미터, 경로

