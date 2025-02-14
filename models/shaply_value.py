import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def explain_global_shap(testloader, combinedModel, age_scaler, bmi_scaler, gender_encoder, side_encoder,
                        presence_encoder, device="cuda"):
    """
    모델이 전체 데이터에 대해 어떤 특성을 중요하게 여기는지 분석하는 Global SHAP 설명 함수.

    Args:
        testloader: 데이터 로더 (torch.utils.data.DataLoader).
        combinedModel: PyTorch 모델.
        age_scaler, bmi_scaler: 연속형 변수 복구용 Scaler.
        gender_encoder, side_encoder, presence_encoder: 범주형 변수 복구용 LabelEncoder.
        device: 실행할 디바이스.

    Returns:
        SHAP Summary Plot을 HTML 파일로 저장.
    """
    combinedModel.eval()
    combinedModel.to(device)

    shap_values_list = []  # 모든 샘플의 SHAP 값을 저장할 리스트
    X_test = []  # 클리닉 데이터를 저장할 리스트

    for batch_idx, (ids, preap_inputs, prelat_inputs, clinic_inputs, labels) in enumerate(testloader):

        preap_inputs = preap_inputs.to(device)
        prelat_inputs = prelat_inputs.to(device)
        clinic_inputs = clinic_inputs.to(device)
        labels = labels.to(device)

        for i in range(clinic_inputs.size(0)):

            # 현재 샘플 추출
            preap_input = preap_inputs[i].unsqueeze(0)  # (1, C, H, W)
            prelat_input = prelat_inputs[i].unsqueeze(0)  # (1, C, H, W)
            clinic_input = clinic_inputs[i].cpu().numpy().reshape(1, -1)  # (1, feature_dim).

            X_test.append(clinic_input)  # 원본 데이터 저장

            # SHAP 예측 함수
            predict_fn = lambda x: shap_predict_fn(x, preap_input, prelat_input, combinedModel, device)

            # SHAP Explainer 생성 및 계산
            explainer = shap.Explainer(predict_fn, clinic_input)
            shap_values = explainer(clinic_input)

            print("shap_values == ", shap_values.shape) #  (1, 5, 2) -> [샘플 개수, 특성개수, 클래스 개수]
            shap_values_class_0 = shap_values.values[..., 0]  # (1, 5, 2) → (1, 5)
            shap_values_class_1 = shap_values.values[..., 1]  # (1, 5, 2) → (1, 5)

            # 개별 샘플의 SHAP 값을 저장
            shap_values_list.append(shap_values.values)  # (1, num_features) 형태

    # 모든 샘플의 SHAP 값을 numpy 배열로 변환
    shap_values_all = np.vstack(shap_values_list)  # (num_samples, num_features)
    global_shap_values = np.mean(shap_values_all, axis=0)  # (num_features,)

    # Global SHAP Summary Plot 저장
    X_test = np.vstack(X_test)  # (전체 샘플 수, feature_dim)
    encoded_shap_image = plot_shap_summary(global_shap_values, X_test, ["age", "bmi", "gender", "side", "presence"])

    # 결과 저장
    save_shap_html(encoded_shap_image, "shap_global_results.html")

def shap_predict_fn(X, preap_input, prelat_input, model, device="cuda"):
    """
    SHAP을 위한 예측 함수 (클리닉 데이터만 사용).

    Args:
        X: 정규화된 임상 데이터 (numpy 배열) (batch_size, num_features)
        preap_input: 단일 샘플의 PreAP 입력 텐서 (1, C, H, W)
        prelat_input: 단일 샘플의 PreLat 입력 텐서 (1, C, H, W)
        model: 훈련된 PyTorch 모델
        device: 실행할 디바이스 (기본값="cuda")

    Returns:
        예측 확률값 (numpy 배열)
    """
    model.to(device)

    # 클리닉 데이터를 pytorch tensor로 변환
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    print("X_tensor==", X_tensor.shape)         # torch.Size([1, 5])
    print("preap_input==", preap_input.shape)   # torch.Size([1, 3, 224, 224])
    print("prelat_input==", prelat_input.shape) # torch.Size([1, 3, 224, 224])

    with torch.no_grad():
        logits = model(preap_input, prelat_input, X_tensor)  # 이미지 + 클리닉 데이터 입력
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


def plot_shap_summary(global_shap_values, X, feature_names):
    """
    SHAP Summary Plot을 생성하여 저장.

    Args:
        global_shap_values: SHAP 분석 결과 (Global SHAP 값, (num_features,))
        X: 입력 데이터 (num_samples, num_features).
        feature_names: 특성 이름 리스트.

    Returns:
        Base64 인코딩된 그래프 이미지.
    """
    num_samples, num_features = X.shape

    # 🔹 Global SHAP 값을 (1, num_features) → (num_samples, num_features) 형태로 변환
    shap_values_expanded = np.tile(global_shap_values.reshape(1, -1), (num_samples, 1))

    # 🔹 차원 확인 (디버깅용)
    print(f"X.shape: {X.shape}, shap_values_expanded.shape: {shap_values_expanded.shape}")

    # 🔹 Summary Plot 생성
    shap.summary_plot(shap_values_expanded, X, feature_names=feature_names, show=False)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return encoded_image




def save_shap_html(encoded_shap_image, file_name):
    """
    SHAP 결과를 HTML 파일로 저장.

    Args:
        encoded_shap_image: Base64 인코딩된 SHAP 그래프 이미지.
        file_name: 저장할 HTML 파일 이름.
    """
    with open(file_name, "w") as f:
        f.write("<html><body><h2>SHAP Feature Importance</h2>\n")
        f.write("<h3>SHAP Summary Plot</h3>\n")
        f.write(f'<img src="data:image/png;base64,{encoded_shap_image}" alt="SHAP Summary">\n')
        f.write("</body></html>")
