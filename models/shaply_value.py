import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def explain_global_shap(testloader, combinedModel, device="cuda"):
    combinedModel.eval()
    combinedModel.to(device)

    shap_values_list = []
    X_test = []

    for batch_idx, (ids, preap_inputs, prelat_inputs, clinic_inputs, labels) in enumerate(testloader):
        preap_inputs = preap_inputs.to(device)
        prelat_inputs = prelat_inputs.to(device)
        clinic_inputs = clinic_inputs.cpu().numpy()

        X_test.append(clinic_inputs)

        for i in range(clinic_inputs.shape[0]):
            preap_input = preap_inputs[i].unsqueeze(0)
            prelat_input = prelat_inputs[i].unsqueeze(0)
            clinic_input = clinic_inputs[i].reshape(1, -1)

            predict_fn = lambda x: shap_predict_fn(x, preap_input, prelat_input, combinedModel, device)

            if batch_idx == 0 and i == 0:
                explainer = shap.Explainer(predict_fn, np.vstack(X_test))

            shap_values = explainer(clinic_input)
            shap_values_list.append(shap_values.values)

    shap_values_all = np.vstack(shap_values_list)
    X_test_np = np.vstack(X_test)

    print("X_test_np shape:", X_test_np.shape)
    print("shap_values_all shape:", shap_values_all.shape)

    # SHAP Summary Plot & Bar Plot 생성
    summary_plot = plot_shap_summary(shap_values_all, X_test_np, ["age", "bmi", "gender", "side", "presence"])
    bar_plot = plot_shap_bar(shap_values_all, ["age", "bmi", "gender", "side", "presence"])

    # HTML 저장
    save_shap_html(summary_plot, bar_plot, "shap_global_results.html")


def shap_predict_fn(clinic_input, preap_input, prelat_input, model, device="cuda"):
    """
    SHAP을 위한 예측 함수 (clinic_input만 변동).

    Args:
        clinic_input: 정규화된 임상 데이터 (numpy 배열) (batch_size, num_features)
        preap_input: 고정된 PreAP 입력 텐서 (1, C, H, W)
        prelat_input: 고정된 PreLat 입력 텐서 (1, C, H, W)
        model: 훈련된 PyTorch 모델
        device: 실행할 디바이스 (기본값="cuda")

    Returns:
        예측 확률값 (numpy 배열)
    """
    model.to(device)

    batch_size = clinic_input.shape[0]

    # clinic_input을 PyTorch Tensor로 변환
    clinic_tensor = torch.tensor(clinic_input, dtype=torch.float32).to(device)

    # preap_input과 prelat_input을 clinic_input의 batch 크기에 맞춰 반복(복제)
    preap_input = preap_input.repeat(batch_size, 1, 1, 1)  # (batch_size, C, H, W)
    prelat_input = prelat_input.repeat(batch_size, 1, 1, 1)  # (batch_size, C, H, W)

    with torch.no_grad():
        logits = model(preap_input, prelat_input, clinic_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


def plot_shap_summary(shap_values, X, feature_names):
    """
    SHAP Summary Plot을 생성하여 저장.

    Args:
        shap_values: SHAP 분석 결과 (Global SHAP 값)
        X: 입력 데이터 (num_samples, num_features).
        feature_names: 특성 이름 리스트.

    Returns:
        Base64 인코딩된 그래프 이미지.
    """
    # print("shap_values shape:", shap_values.shape)  # 디버깅
    # print("X shape:", X.shape)  # 디버깅
    # print("Feature names:", feature_names)  # 디버깅

    # feature_names를 numpy 배열로 변환 (인덱싱 오류 방지)
    feature_names = np.array(feature_names)

    # SHAP 값이 다중 클래스일 경우, 평균을 내어 차원을 줄임
    if len(shap_values.shape) == 3:
        shap_values = np.mean(shap_values, axis=2)  # (num_samples, num_features)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names.tolist(), show=False)

    shap_range = np.max(np.abs(shap_values))  # 최댓값을 기반으로 설정
    plt.xlim(-shap_range, shap_range)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return encoded_image

def plot_shap_bar(shap_values, feature_names):
    """
    SHAP Bar Plot을 생성하여 저장.

    Args:
        shap_values: SHAP 분석 결과 (Global SHAP 값)
        feature_names: 특성 이름 리스트.

    Returns:
        Base64 인코딩된 그래프 이미지.
    """
    plt.figure()

    # SHAP 값이 다중 클래스일 경우, 평균을 내어 차원을 줄임
    if len(shap_values.shape) == 3:
        shap_values = np.mean(shap_values, axis=2)  # (num_samples, num_features)

    # 평균 절대 SHAP 값 계산
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Bar plot 생성
    shap.bar_plot(mean_shap_values, feature_names)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return encoded_image


def save_shap_html(summary_plot, bar_plot, file_name):
    """
    SHAP 결과를 HTML 파일로 저장.

    Args:
        summary_plot: Base64 인코딩된 SHAP Summary Plot 이미지.
        bar_plot: Base64 인코딩된 SHAP Bar Plot 이미지.
        file_name: 저장할 HTML 파일 이름.
    """
    with open(file_name, "w") as f:
        f.write("<html><body><h2>SHAP Feature Importance</h2>\n")
        f.write("<h3>SHAP Summary Plot</h3>\n")
        f.write(f'<img src="data:image/png;base64,{summary_plot}" alt="SHAP Summary">\n')
        f.write("<h3>SHAP Bar Plot (Feature Importance Ranking)</h3>\n")
        f.write(f'<img src="data:image/png;base64,{bar_plot}" alt="SHAP Bar Plot">\n')
        f.write("</body></html>")

