import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def explain_global_shap(testloader, combinedModel, device="cuda"):
    """
    SHAP을 이용해 clinic_input의 feature 중요도를 분석하는 Global SHAP 설명 함수.
    - 클래스 0과 1 각각의 SHAP Bar Plot 생성
    - 클래스 0과 1을 통합한 SHAP Bar Plot 추가

    Args:
        testloader: 데이터 로더 (torch.utils.data.DataLoader).
        combinedModel: PyTorch 모델.
        device: 실행할 디바이스.

    Returns:
        SHAP Summary Plot, 각 클래스별 SHAP Bar Plot, 통합 SHAP Bar Plot을 포함한 HTML 파일 저장.
    """
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

    feature_names = ["age", "bmi", "gender", "side", "presence"]

    # SHAP Summary Plot 생성
    summary_plot = plot_shap_summary(shap_values_all, X_test_np, feature_names)

    # 📌 클래스 0과 1 각각의 SHAP Bar Plot 생성
    class_bar_plots = plot_shap_bar_multiclass(shap_values_all, feature_names)

    # 📌 클래스 0과 1을 통합한 SHAP Bar Plot 생성
    combined_bar_plot = plot_shap_bar_combined(shap_values_all, feature_names)

    # HTML 저장
    save_shap_html(summary_plot, class_bar_plots, combined_bar_plot, "shap_global_results.html")




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
    plt.figure()

    if len(shap_values.shape) == 3:
        shap_values = np.mean(shap_values, axis=2)  # (num_samples, num_features)

    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return encoded_image

def plot_shap_bar_combined(shap_values, feature_names):
    """
    클래스 0과 1을 통합한 SHAP Bar Plot 생성.

    Args:
        shap_values: SHAP 분석 결과 (num_samples, num_features, num_classes)
        feature_names: 특성 이름 리스트.

    Returns:
        Base64 인코딩된 그래프 이미지.
    """
    plt.figure(figsize=(10, 6))

    # 📌 클래스별 평균 절대 SHAP 값을 합산하여 전체 중요도를 나타냄
    combined_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))

    plt.barh(feature_names, combined_shap_values, color='purple')
    plt.xlabel("mean(|SHAP value|) (Aggregated over Classes 0 & 1)")
    plt.title("Combined Feature Importance (Classes 0 & 1)")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return encoded_image


def plot_shap_bar_multiclass(shap_values, feature_names):
    """
    다중 클래스(클래스 0과 1) SHAP Bar Plot을 각각 생성.

    Args:
        shap_values: SHAP 분석 결과 (num_samples, num_features, num_classes)
        feature_names: 특성 이름 리스트.

    Returns:
        Base64 인코딩된 그래프 이미지 리스트.
    """
    encoded_images = []

    num_classes = shap_values.shape[2]  # 다중 클래스 개수

    for class_idx in range(num_classes):
        plt.figure(figsize=(10, 6))

        class_shap_values = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)

        plt.barh(feature_names, class_shap_values, color='red')
        plt.xlabel(f"mean(|SHAP value|) for Class {class_idx}")
        plt.title(f"Feature Importance for Class {class_idx}")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        buf.seek(0)
        encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        encoded_images.append(encoded_image)

    return encoded_images


def save_shap_html(summary_plot, class_bar_plots, combined_bar_plot, file_name):
    """
    SHAP 결과를 HTML 파일로 저장.

    Args:
        summary_plot: Base64 인코딩된 SHAP Summary Plot 이미지.
        class_bar_plots: Base64 인코딩된 각 클래스별 SHAP Bar Plot 이미지 리스트.
        combined_bar_plot: Base64 인코딩된 통합 SHAP Bar Plot 이미지.
        file_name: 저장할 HTML 파일 이름.
    """
    with open(file_name, "w") as f:
        f.write("<html><body><h2>SHAP Feature Importance</h2>\n")

        f.write("<h3>SHAP Summary Plot</h3>\n")
        f.write(f'<img src="data:image/png;base64,{summary_plot}" alt="SHAP Summary" style="width:100%;">\n')

        # 클래스 0과 1 각각의 SHAP Bar Plot 추가
        for class_idx, bar_plot in enumerate(class_bar_plots):
            f.write(f"<h3>SHAP Bar Plot (Feature Importance for Class {class_idx})</h3>\n")
            f.write(f'<img src="data:image/png;base64,{bar_plot}" alt="SHAP Bar Plot for Class {class_idx}" style="width:100%;">\n')

        #  클래스 0과 1을 통합한 SHAP Bar Plot 추가
        f.write("<h3>SHAP Bar Plot (Aggregated Feature Importance for Classes 0 & 1)</h3>\n")
        f.write(f'<img src="data:image/png;base64,{combined_bar_plot}" alt="Combined SHAP Bar Plot" style="width:100%;">\n')

        f.write("</body></html>")




