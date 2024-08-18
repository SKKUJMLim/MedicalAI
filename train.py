from tqdm.auto import tqdm
import torch

# 모델을 학습시키는 함수 작성
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
            epoch_corrects  = 0  # epoch 정답 수

            # 학습하지 않을 시 검증 성능을 확인하기 위해 epoch=0의 훈련 생략
            if (epoch == 0) and (phase == 'train'):
                continue

            # Dataloader로 미니 배치를 꺼내는 루프
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPU가 사용가능하면 GPU에 데이터 전송
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 옵티마이저 초기화
                optimizer.zero_grad()

                # 순전파 계산
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = net(inputs)
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

    return net