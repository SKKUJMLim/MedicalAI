import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import get_dataloader
from models import combinedModel, U_Net, clinicinfo, gradcam,lime, gradcamforimage 
from lime.lime_tabular import LimeTabularExplainer
import gc

parser = argparse.ArgumentParser(description='Test a medical image classifier.')
parser.add_argument('--model_name', type=str, default='notpreMedicalNet', help='Model name to load')


args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    resize = 512
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 1
    num_classes = 2

    # Load test data
    _, test_dataloader = get_dataloader(resize, mean, std, batch_size)
    
    # Load model components
    preap_net = U_Net.UNet()
    prelat_net = U_Net.UNet()
    clinicinfo_net = clinicinfo.MLP(input_size=2, hidden_size=2, output_size=1)
    
    # Define combined model
    class UNetEncoder(nn.Module):
        def __init__(self, unet):
            super(UNetEncoder, self).__init__()
            self.unet = unet
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512 * 512, 512)  # Match the dimension as per model design

        def forward(self, x):
            x = self.unet(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    preap_net = UNetEncoder(preap_net)
    prelat_net = UNetEncoder(prelat_net)
    combined_model = combinedModel.CombinedUnet(preap_net, prelat_net, clinicinfo_net, num_classes)
    #combined_model = combinedModel.CombinedResNet18_onlyImage(preap_net, prelat_net, num_classes)

    # Load the best model
    #combined_model.load_state_dict(torch.load(r"/sehun/medicalai/medical_server/best_model.pth", map_location=device))
    combined_model.load_state_dict(torch.load(r"best 모델 경로 설정정", map_location=device, weights_only = True))
    combined_model.to(device)
    combined_model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_scores = [] #양성 클래스 확률 저장

    with torch.no_grad():
        for ids,preap_inputs, prelat_inputs, clinic_inputs, labels in test_dataloader:
            preap_inputs = preap_inputs.to(device)
            prelat_inputs = prelat_inputs.to(device)
            clinic_inputs = clinic_inputs.to(device)
            labels = labels.to(device)
            outputs = combined_model(preap_inputs, prelat_inputs, clinic_inputs)
            #outputs = combined_model(preap_inputs, prelat_inputs)

            #양성 클래스 확률
            probs = torch.softmax(outputs,dim=1)[:,1].cpu().numpy()
            all_scores.extend(probs)


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Accuracy calculation
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    # Confusion Matrix visualization
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for MedicalAI')
    plt.savefig('confusion_matrix_test.png')
    plt.show()

    fpr,tpr,thresholds = roc_curve(all_labels,all_scores,pos_label=1)
    roc_auc = auc(fpr,tpr)
    print(f"AUC:{roc_auc:.4f}")
    # ROC 커브 시각화
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

    '''Grad-CAM'''
    # 정면 이미지를 위한 Grad-CAM
    grad_cam = gradcam.GradCAM(model=combined_model.model1, target_layer=combined_model.model1.unet.bottleneck)
    gradcam.save_all_grad_cam_results(grad_cam=grad_cam, image_type='preap' , model=combined_model.model1, testloader=test_dataloader,combinedModel=combined_model)

    ## 측면 이미지를 위한 Grad-CAM
    grad_cam = gradcam.GradCAM(model=combined_model.model2, target_layer=combined_model.model2.unet.bottleneck)
    gradcam.save_all_grad_cam_results(grad_cam=grad_cam, image_type='prelat', model=combined_model.model2, testloader=test_dataloader,combinedModel=combined_model)
    

    
    
    torch.cuda.empty_cache()

    gc.collect()
    '''LIME'''

    #모델 cpu 이동
    # combined_model.cpu()
    # device = 'cpu'

    training_data = []
    for _, (_, _, _, clinic_inputs, _) in enumerate(test_dataloader):
        
        training_data.append(clinic_inputs.cpu().numpy())
    training_data = np.vstack(training_data)

    # LimeTabularExplainer 초기화
    explainer = LimeTabularExplainer(
        training_data=training_data,  # 훈련 데이터 (numpy 배열)
        feature_names=['age,', 'bmi'],  # 특성 이름 (리스트)
        class_names=[0, 1],  # 클래스 이름 (리스트)
        mode="classification"  # 모드: 분류(classification) 또는 회귀(regression)
    )

    # 설명 생성
    explanations = lime.explain_instance(test_dataloader, explainer, combined_model, device=device)
    lime.save_all_lime_results(explanations)