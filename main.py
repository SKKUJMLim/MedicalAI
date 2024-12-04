from dataloader import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from models import resnet, vgg, combinedModel, clinicinfo
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models import gradcam
import utils

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 장치:", device)

    # 난수 시드 고정
    torch.manual_seed(42)

    resize = 224
    # resize = 32
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 2
    batch_size = 8
    num_epochs = 1
    learning_rate = 0.001

    best_accuracy = 0.0
    best_model_path = './best_model.pth'

    train_loaders_dict, test_dataloader = get_dataloader(resize, mean, std, batch_size)

    print('==> Building model..')
    # resnet.test()
    preap_net = resnet.resnet34(pretrained=True)
    prelat_net= resnet.resnet34(pretrained=True)
    clinicinfo_net = clinicinfo.MLP(input_size=2, hidden_size=2, output_size=1)
    combined_model = combinedModel.CombinedResNet18(preap_net, prelat_net, clinicinfo_net, num_classes)

    # vgg.test()
    # preap_net = vgg.VGG('VGG19')
    # prelat_net = vgg.VGG('VGG19')
    # combined_model = combinedModel.CombinedVGG(preap_net, prelat_net, num_classes)

    combined_model.to(device=device)

    # 최적화 기법 설정
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(combined_model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    # optimizer = optim.Adam([
    #     {'params': preap_net.parameters(), 'lr':learning_rate},
    #     {'params': prelat_net.parameters(), 'lr':learning_rate},
    # ])

    # 학습 및 검증 실시


    ## epoch 루프
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print("------------------------------")

        running_loss = 0.0
        combined_model.train()

        trainloader =  train_loaders_dict['train']

        for ids, preap_inputs, prelat_inputs, clinic_inputs, labels in trainloader:

            # GPU가 사용가능하면 GPU에 데이터 전송
            preap_inputs = preap_inputs.to(device)
            prelat_inputs = prelat_inputs.to(device)
            clinic_inputs = clinic_inputs.to(device)
            labels = labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()
            outputs = combined_model(preap_inputs, prelat_inputs, clinic_inputs)

            '''1. 단순 cross-enropy loss'''
            loss = criterion(outputs, labels)

            '''2. Focal loss'''
            # Initialize Focal Loss
            # focal_loss = utils.FocalLoss(alpha=1, gamma=2, reduction='mean')
            # loss = focal_loss(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(trainloader)}')

        # Validation
        combined_model.eval()
        correct = 0
        total = 0
        validationloader = train_loaders_dict['val']

        with torch.no_grad():
            for ids, preap_inputs, prelat_inputs, clinic_inputs, labels in validationloader:
                preap_inputs = preap_inputs.to(device)
                prelat_inputs = prelat_inputs.to(device)
                clinic_inputs = clinic_inputs.to(device)
                labels = labels.to(device)
                outputs = combined_model(preap_inputs, prelat_inputs, clinic_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

        # Best 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
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
        for ids, preap_inputs, prelat_inputs, clinic_inputs, labels in test_dataloader:
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

    '''Confusion Matrix 생성'''
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
    grad_cam = gradcam.GradCAM(model=combined_model.model1, target_layer=combined_model.model1.layer4)
    gradcam.save_all_grad_cam_results(grad_cam=grad_cam, image_type='preap' , model=combined_model.model1, testloader=test_dataloader, combinedModel=combined_model)

    ## 측면 이미지를 위한 Grad-CAM
    grad_cam = gradcam.GradCAM(model=combined_model.model2, target_layer=combined_model.model2.layer4)
    gradcam.save_all_grad_cam_results(grad_cam=grad_cam, image_type='prelat', model=combined_model.model2, testloader=test_dataloader, combinedModel=combined_model)

