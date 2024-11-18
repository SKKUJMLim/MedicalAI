import argparse
from dataloader import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from models import resnet, vgg, combinedModel, clinicinfo
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from models import gradcam
from models import U_Net


parser = argparse.ArgumentParser(description = 'Train a medical image classifier.')
parser.add_argument('--model_name', type=str, default='notpreMedicalNet')

args = parser.parse_args()
model_name = args.model_name

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 장치:", device)

    # 난수 시드 고정
    torch.manual_seed(42) #토치 시드 외 파이썬 시드 고정

    # resize = 224
    resize = 512
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 2
    batch_size = 4
    num_epochs = 100
    learning_rate = 0.001

    best_accuracy = 0.0
    best_model_path = f'./{model_name}unprebest_model.pth'

    train_loaders_dict, test_dataloader = get_dataloader(resize, mean, std, batch_size)
    #train_loaders_dict, test_dataloader = get_dataloader(resize, mean, std, batch_size)
    print('==> Building model..')
    # resnet.test()
    preap_net = U_Net.UNet()
    prelat_net = U_Net.UNet()
    clinicinfo_net = clinicinfo.MLP(input_size=2, hidden_size=2, output_size=1)
    
    #U-Net의 출력 크기를 조정하기 위해 Flatten 및 Linear 레이어 추가
    class UNetEncoder(nn.Module):
        def __init__(self, unet):
            super(UNetEncoder, self).__init__()
            self.unet = unet
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512 * 512, 512)# 임의로 설정한 출력 차원 크기

        def forward(self, x):
            x = self.unet(x)
            x = self.flatten(x)
            #print('flatten',x.shape)
            x = self.fc(x)
            #print('fc', x.shape)
            return x
    preap_net = UNetEncoder(preap_net)
    prelat_net = UNetEncoder(prelat_net)
    #print(preap_net)

    combined_model = combinedModel.CombinedUnet(preap_net, prelat_net, clinicinfo_net, num_classes)

    # vgg.test()
    # preap_net = vgg.VGG('VGG19')
    # prelat_net = vgg.VGG('VGG19')
    # combined_model = combinedModel.CombinedVGG(preap_net, prelat_net, num_classes)

    combined_model.to(device=device)

    # 최적화 기법 설정
    criterion = nn.CrossEntropyLoss()
    
    # optimizer = optim.SGD(combined_model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)
    log_dir = f'./tensorboard_logs/{model_name}'
    writer = SummaryWriter(log_dir=log_dir)
    # optimizer = optim.Adam([
    #     {'params': preap_net.parameters(), 'lr':learning_rate},
    #     {'params': prelat_net.parameters(), 'lr':learning_rate},
    # ])

    # 학습 및 검증 실시
    print()

    ## epoch 루프
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print("------------------------------")

        running_loss = 0.0
        correct = 0
        total = 0
        combined_model.train()

        trainloader =  train_loaders_dict['train']

        for preap_inputs, prelat_inputs, clinic_inputs, labels in trainloader:

            # GPU가 사용가능하면 GPU에 데이터 전송
            preap_inputs = preap_inputs.to(device)
            prelat_inputs = prelat_inputs.to(device)
            clinic_inputs = clinic_inputs.to(device)
            labels = labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()
            outputs = combined_model(preap_inputs, prelat_inputs, clinic_inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Accuracy calculation during training
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(trainloader)
        train_accuracy = 100* correct / total

        writer.add_scalar('Training Loss', train_loss, epoch + 1)
        writer.add_scalar('Training Accuracy', train_accuracy, epoch + 1)
        print(f'Epoch [{epoch + 1}/{num_epochs}] traning Loss: {train_loss:.4f} Training accuracy: {train_accuracy:.2f}% ')

        # Validation
        combined_model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        validationloader = train_loaders_dict['val']

        with torch.no_grad():
            for preap_inputs, prelat_inputs, clinic_inputs, labels in validationloader:
                preap_inputs = preap_inputs.to(device)
                prelat_inputs = prelat_inputs.to(device)
                clinic_inputs = clinic_inputs.to(device)
                labels = labels.to(device)
                outputs = combined_model(preap_inputs, prelat_inputs, clinic_inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = running_val_loss / len(validationloader)
        val_accuracy = 100 * correct / total
        # Log validation metrics to TensorBoard
        writer.add_scalar('Validation Loss', val_loss, epoch + 1)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch + 1)

        print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_accuracy:.2f}%')



        # Best 모델 저장
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(combined_model.state_dict(), best_model_path)
            print(f'Saving model with accuracy: {best_accuracy}%')


    # 테스트 데이터셋에 대한 성능 평가
    combined_model.load_state_dict(torch.load(best_model_path))
    combined_model.eval()
    correct = 0
    total = 0

    # 실제 라벨과 예측 라벨을 저장할 리스트 (Confusion matrix)
    all_labels = []
    all_preds = []


    with torch.no_grad():
        for preap_inputs, prelat_inputs, clinic_inputs, labels in test_dataloader:
            # GPU가 사용가능하면 GPU에 데이터 전송
            preap_inputs = preap_inputs.to(device)
            prelat_inputs = prelat_inputs.to(device)
            clinic_inputs = clinic_inputs.to(device)
            labels = labels.to(device)
            combined_model.to(device)
            outputs = combined_model(preap_inputs, prelat_inputs, clinic_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())  # 실제 라벨 저장
            all_preds.extend(predicted.cpu().numpy())  # 예측 라벨 저장
    print(f'Accuracy of the best model on the test images: {100 * correct / total}%')

    # Confusion Matrix 생성
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Confusion Matrix 시각화 및 이미지 저장
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for MedicalAI')

    # 이미지 파일로 저장 (PNG 형식)
    plt.savefig('confusion_matrix.png')

    '''Grad-CAM'''
    ## 정면 이미지를 위한 Grad-CAM
    grad_cam = gradcam.GradCAM(model=combined_model.model1, target_layer=combined_model.model1.bottleneck)
    gradcam.save_all_grad_cam_results(grad_cam=grad_cam, image_type='preap' , model=combined_model.model1, testloader=test_dataloader)

    ## 측면 이미지를 위한 Grad-CAM
    grad_cam = gradcam.GradCAM(model=combined_model.model2, target_layer=combined_model.model2.bottleneck)
    gradcam.save_all_grad_cam_results(grad_cam=grad_cam, image_type='prelat', model=combined_model.model2, testloader=test_dataloader)
   
    writer.add_scalar('Test Accuracy', val_accuracy)

    # SummaryWriter 닫기
    writer.close()