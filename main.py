from dataloader import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from models import resnet, vgg, combinedModel, clinicinfo
from sklearn.metrics import confusion_matrix
import seaborn as sns
from models import gradcam

from lime.lime_tabular import LimeTabularExplainer
from models import lime, shaply_value
import numpy as np
import matplotlib.pyplot as plt
import os


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
    clinicinfo_net = clinicinfo.MLP(input_size=5, hidden_size=3, output_size=1)
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

    # '''Grad-CAM'''
    # ## 정면 이미지를 위한 Grad-CAM
    # grad_cam = gradcam.GradCAM(model=combined_model.model1, target_layer=combined_model.model1.layer4)
    # gradcam.save_all_grad_cam_results(grad_cam=grad_cam, image_type='preap' , model=combined_model.model1, testloader=test_dataloader, combinedModel=combined_model)
    #
    # ## 측면 이미지를 위한 Grad-CAM
    # grad_cam = gradcam.GradCAM(model=combined_model.model2, target_layer=combined_model.model2.layer4)
    # gradcam.save_all_grad_cam_results(grad_cam=grad_cam, image_type='prelat', model=combined_model.model2, testloader=test_dataloader, combinedModel=combined_model)

    '''LIME'''
    # training_data = []
    # for _, (_, _, _, clinic_inputs, _) in enumerate(test_dataloader):
    #     training_data.append(clinic_inputs.cpu().numpy())
    #
    # training_data = np.vstack(training_data)
    # age_scaler, bmi_scaler, gender_encoder, side_encoder, presence_encoder = test_dataloader.dataset.get_scaler()
    #
    # # 범주형 Feature 인덱스 설정
    # categorical_features = [2, 3, 4]  # gender(2), side(3), presence(4)
    #
    # training_data[:, categorical_features] = training_data[:, categorical_features].astype(int)
    #
    # # 범주형 변수의 원래 값 설정 (LabelEncoder 사용)
    # categorical_names = {
    #     2: gender_encoder.classes_.tolist(),  # LabelEncoder에서 직접 클래스 목록 가져오기
    #     3: side_encoder.classes_.tolist(),    # LabelEncoder에서 직접 클래스 목록 가져오기
    #     4: presence_encoder.classes_.tolist() # LabelEncoder에서 직접 클래스 목록 가져오기
    # }
    #
    # # LimeTabularExplainer 초기화
    # explainer = LimeTabularExplainer(
    #     training_data=training_data,
    #     feature_names=['age', 'bmi', 'gender', 'side', 'presence'],
    #     class_names=[0, 1],
    #     categorical_features=categorical_features,  # 범주형 Feature 인덱스 설정
    #     categorical_names=categorical_names,  # 범주형 변수의 원래 값 설정
    #     mode="classification"
    # )
    #
    # # 설명 생성
    # explanations = lime.explain_instance(test_dataloader, explainer, combined_model, device='cuda', max_samples=5000)
    # lime.save_all_lime_results(explanations,
    #                            age_scaler=age_scaler,
    #                            bmi_scaler=bmi_scaler,
    #                            gender_encoder=gender_encoder,
    #                            side_encoder=side_encoder,
    #                            presence_encoder=presence_encoder)

    '''Shaply value'''
    # 정규화 스케일러 가져오기
    age_scaler, bmi_scaler, gender_encoder, side_encoder, presence_encoder = test_dataloader.dataset.get_scaler()

    # Global SHAP 실행 (전체 데이터 설명)
    shaply_value.explain_global_shap(test_dataloader, combined_model, age_scaler, bmi_scaler, gender_encoder,
                                     side_encoder,
                                     presence_encoder, device='cuda')


    # # 설명할 Feature와 이미지 데이터를 저장할 리스트 초기화
    # X_test = []         # 설명할 Feature (임상 데이터)
    # preap_inputs = []   # AP 이미지
    # prelat_inputs = []  # LAT 이미지
    #

    #
    # for _, (_, preap_input, prelat_input, clinic_inputs, _) in enumerate(test_dataloader):
    #     X_test.append(clinic_inputs.cpu().numpy())  # 임상 데이터 저장
    #     preap_inputs.append(preap_input.cpu().numpy())  # AP 이미지 저장
    #     prelat_inputs.append(prelat_input.cpu().numpy())  # LAT 이미지 저장
    #
    # # 리스트를 numpy 배열로 변환
    # X_test = np.vstack(X_test)
    # preap_inputs = np.vstack(preap_inputs)
    # prelat_inputs = np.vstack(prelat_inputs)
    #
    # # 텐서 변환 후 GPU로 이동
    # preap_inputs = torch.tensor(preap_inputs, dtype=torch.float32).to("cuda")  # (228, C, H, W)
    # prelat_inputs = torch.tensor(prelat_inputs, dtype=torch.float32).to("cuda")  # (228, C, H, W)

    # Local SHAP 실행 (샘플별 설명)
    # shaply_value.explain_instance_with_shap(test_dataloader, combined_model, age_scaler, bmi_scaler, gender_encoder, side_encoder,
    #                            presence_encoder, device='cuda', max_samples=20)





