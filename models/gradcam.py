import torch
import numpy as np
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import os

# Grad-CAM 결과 계산 및 저장
def save_all_grad_cam_results(grad_cam, model, testloader, image_type, combinedModel ,device='cuda'):

    # Grad-CAM 결과 저장 폴더 생성
    os.makedirs(f'grad_cam_results/{image_type}/0/Correct', exist_ok=True)
    os.makedirs(f'grad_cam_results/{image_type}/0/Incorrect', exist_ok=True)
    os.makedirs(f'grad_cam_results/{image_type}/1/Correct', exist_ok=True)
    os.makedirs(f'grad_cam_results/{image_type}/1/Incorrect', exist_ok=True)

    model.eval()  # 모델을 평가 모드로 전환

    for batch_idx, (ids, preap_inputs, prelat_inputs, clinic_inputs, labels) in tqdm(enumerate(testloader), total=len(testloader), desc="Calculating Grad-CAM"):


        if image_type == 'preap':
            images = preap_inputs
        elif image_type == 'prelat':
            images = prelat_inputs
        else:
            print("GradCAM.. Image type Error")


        images, labels = images.to(device), labels.to(device)

        # 배치의 각 이미지에 대해 Grad-CAM 계산
        for i in range(images.size(0)):
            image = images[i].unsqueeze(0)  # 이미지 추출 및 배치 차원 유지
            label = labels[i].item()  # 타겟 클래스 가져오기
            cam = grad_cam(image, label)  # Grad-CAM 결과 계산

            # Grad-CAM 결과를 이미지 형태로 변환
            cam_resized = cv2.resize(cam, (image.shape[2], image.shape[3]))  # 원본 이미지 크기로 조정
            cam_image = cam_resized - cam_resized.min()
            cam_image = cam_image / cam_image.max() * 255.0  # 스케일 조정 및 uint8로 변환
            cam_image = cam_image.astype(np.uint8)

            # 원본 이미지와 Grad-CAM 히트맵을 오버레이하여 저장
            original_image = images[i].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
            original_image = (original_image - original_image.min()) / (
                        original_image.max() - original_image.min()) * 255.0
            original_image = original_image.astype(np.uint8)
            overlay = cv2.addWeighted(original_image, 0.6, cv2.applyColorMap(cam_image, cv2.COLORMAP_JET), 0.4, 0)


            ##
            id = ids[i]
            preap_input = preap_inputs[i].to(device).unsqueeze(0)
            prelat_input = prelat_inputs[i].to(device).unsqueeze(0)
            clinic_input = clinic_inputs[i].to(device).unsqueeze(0)
            output = combinedModel(preap_input, prelat_input, clinic_input)
            _, predicted = torch.max(output.data, 1)

            if predicted == label:
                prediction = 'Correct'
            else:
                prediction = 'Incorrect'

            filename = f"grad_cam_results/{image_type}/{label}/{prediction}/{id}.png"
            cv2.imwrite(filename, overlay)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, target_class):
        # Forward pass
        output = self.model(x)
        self.model.zero_grad()

        # 특정 클래스에 대해 backward pass 실행
        # :는 배치의 모든 샘플을 의미
        loss = output[:, target_class].sum()
        loss.backward()

        # 활성화 맵과 그레이디언트를 가져옴
        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Grad-CAM 계산
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU 적용

        # 해상도 조정 및 정규화
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()  # (224, 224)로 변환

        # 정규화
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 0~1로 정규화
        return cam



