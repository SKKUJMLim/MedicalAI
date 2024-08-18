from dataloader import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from models import resnet, vgg, combinedModel


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 난수 시드 고정
    torch.manual_seed(42)

    resize = 32
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 2
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    train_loaders_dict, test_data = get_dataloader(resize, mean, std, batch_size)

    print('==> Building model..')
    preap_net = resnet.ResNet50()
    # preap_net.to(device)

    prelat_net= resnet.ResNet50()
    # prelat_net.to(device)

    # CombinedResNet 모델 초기화
    combined_model = combinedModel.CombinedResNet50(preap_net, prelat_net, num_classes)
    combined_model.to(device=device)

    # 최적화 기법 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    # optimizer = optim.Adam([
    #     {'params': preap_net.parameters(), 'lr':learning_rate},
    #     {'params': prelat_net.parameters(), 'lr':learning_rate},
    # ])

    # 학습 및 검증 실시
    def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("사용 장치:", device)

        ## epoch 루프
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print("------------------------------")

            net.to(device)

            # epoch 별 학습 및 검증 루프
            for phase in ['train', 'val']:

                if phase == 'train':
                    net.train()  # 모델을 훈련 모드로
                else:
                    net.eval()  # 모델을 검증 모드로

                epoch_loss = 0.0  # epcch 손실 합
                epoch_corrects = 0  # epoch 정답 수

                # 학습하지 않을 시 검증 성능을 확인하기 위해 epoch=0의 훈련 생략
                if (epoch == 0) and (phase == 'train'):
                    continue

                # Dataloader로 미니 배치를 꺼내는 루프

                print("dataloaders_dict[phase] == ", dataloaders_dict[phase]['preap'])
                for inputs, labels in dataloaders_dict[phase]:

                    print("inputs == ", inputs)

                    # GPU가 사용가능하면 GPU에 데이터 전송
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 옵티마이저 초기화
                    optimizer.zero_grad()

                    # 순전파 계산
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = net(inputs, inputs)
                        loss = criterion(outputs, labels)  # 손실 계산
                        _, preds = torch.max(outputs, 1)  # 라벨 예측

                        # 훈련 시에는 오차 역전파
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # 반복 결과 계산
                        # 손실 합계 갱신
                        epoch_loss += loss.item() * inputs.size(0)
                        # 정답 수의 합계 갱신
                        epoch_corrects += torch.sum(preds == labels.data)

                        # epcch 당 손실과 정답률 표시
                        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

                        print('{} Loss: {:.4f} Acc : {:.4f}'.format(phase, epoch_loss, epoch_acc))


    train_model(combined_model, train_loaders_dict, criterion, optimizer, num_epochs=num_epochs)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            images, labels = data

            preap_net.eval()

            # GPU가 사용가능하면 GPU에 데이터 전송
            preap_net.to(device)
            images = images.to(device)
            labels = labels.to(device)

            outputs = preap_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    save_path = './weights_transfer_learning.pth'
    torch.save(preap_net.state_dict(), save_path)  # 저장할 파라미터, 경로