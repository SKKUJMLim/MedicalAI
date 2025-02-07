import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

num_samples = 20  # perturbation 샘플 개수


def explain_with_original_data_and_ranges(explanation, age_scaler, bmi_scaler, gender_encoder, side_encoder,
                                          presence_encoder):
    """
    정규화된 데이터로 모델 예측, 원본 값으로 설명 기준을 변환.

    Args:
        explanation: 정규화된 데이터에 대한 LIME 결과.
        age_scaler, bmi_scaler: 정규화된 연속형 변수의 Scaler 객체.
        gender_encoder, side_encoder, presence_encoder: 범주형 변수 복원을 위한 LabelEncoder 객체.

    Returns:
        변환된 LIME 설명 리스트 (feature, 원래값, 변환된값, weight 형태).
    """
    import re

    # **원래 feature 값 가져오기**
    normalized_instance = np.array(explanation.domain_mapper.feature_values).reshape(1, -1)  # (1, feature_dim)

    # 연속형 변수(Age, BMI)는 Scaler를 사용하여 원래 값으로 변환
    original_values = [
        age_scaler.inverse_transform(normalized_instance[:, [0]].astype(float))[0][0],  # Age
        bmi_scaler.inverse_transform(normalized_instance[:, [1]].astype(float))[0][0]  # BMI
    ]

    # 문자열에서 숫자만 추출하는 함수
    def extract_float(value):
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        return float(match.group()) if match else None

    # 연속형 변수 변환 함수
    def transform_number(value, scaler, feature_name):
        """
        부등호와 숫자를 개별적으로 변환하여 원래 값을 복원.

        Args:
            value: LIME이 생성한 feature string (예: "0.28 < bmi <= 0.37", "bmi > 0.37").
            scaler: 변환할 Scaler 객체.
            feature_name: 변환할 feature 이름 (예: 'age', 'bmi').

        Returns:
            변환된 값 (예: "23.79 < bmi <= 26.64" 또는 "bmi > 26.64").
        """
        # 1. 범위 조건 패턴 (0.28 < bmi <= 0.37)
        range_pattern = r"([-+]?\d*\.\d+|\d+)\s*([<>]=?)\s*(\w+)\s*([<>]=?)\s*([-+]?\d*\.\d+|\d+)"
        range_match = re.search(range_pattern, value)

        if range_match:
            lower_num, lower_op, var, upper_op, upper_num = range_match.groups()
            lower_bound = scaler.inverse_transform([[float(lower_num)]])[0][0]
            upper_bound = scaler.inverse_transform([[float(upper_num)]])[0][0]
            return f"{lower_bound:.2f} {lower_op} {var} {upper_op} {upper_bound:.2f}"

        # 2. 단일 조건 패턴 (bmi <= 0.37)
        single_pattern = r"(\w+)\s*([<>]=?)\s*([-+]?\d*\.\d+|\d+)"
        single_match = re.search(single_pattern, value)

        if single_match:
            var, op, num = single_match.groups()
            transformed_value = scaler.inverse_transform([[float(num)]])[0][0]
            return f"{var} {op} {transformed_value:.2f}"

        return value

    # 변환된 설명 리스트
    transformed_explanation = []

    # 설명 수정: 정규화된 기준을 원본 값으로 변환
    for feature, weight in explanation.as_list():
        original_value = None  # 원래 feature 값
        transformed_value = None  # 변환된 값

        if 'age' in feature:
            scaler = age_scaler
            original_value = original_values[0]  # Age 원래 값
            transformed_value = transform_number(feature, scaler, 'age')

        elif 'bmi' in feature:
            scaler = bmi_scaler
            original_value = original_values[1]  # BMI 원래 값
            transformed_value = transform_number(feature, scaler, 'bmi')
        elif 'gender' in feature:
            original_value = "Male" if '1' in feature else "Female"
            transformed_value = f"{feature.split('=')[-1]} ({original_value})"

        elif 'side' in feature:
            original_value = "Right" if '1' in feature else "Left"
            transformed_value = f"{feature.split('=')[-1]} ({original_value})"

        elif 'presence' in feature:
            original_value = "Subsequent x" if '0' in feature else "Subsequent o"
            transformed_value = f"{feature.split('=')[-1]} ({original_value})"
        else:
            transformed_explanation.append((feature, None, None, weight))
            continue

        # 변환된 feature 추가
        transformed_explanation.append((feature, original_value, transformed_value, weight))

    return transformed_explanation


def save_transformed_explanation_html(transformed_explanation, file_name, prediction_probabilities):
    """
    HTML 파일에 변환된 LIME 설명 및 상대적 중요도 그래프 저장.
    """

    # **1. Prediction Probabilities 그래프 생성**
    fig, ax = plt.subplots(figsize=(3, 2))
    classes = [f"Class {i}" for i in range(len(prediction_probabilities))]
    bars = ax.bar(classes, prediction_probabilities, color=['blue', 'orange'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")

    # **1.1 막대 위에 클래스명 추가**
    for bar, class_name in zip(bars, classes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, class_name, ha='center', fontsize=10,
                fontweight='bold')

    # **2. Prediction Probabilities 그래프 → Base64 변환**
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_prob_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    # **3. clinic_inputs Feature Importance 계산**
    feature_importance = compute_relative_importance(transformed_explanation)

    # **3.1 각 Feature가 Class 0인지 Class 1에 기여하는지 추출 (올바르게 수정된 부분)**
    feature_contributions = {}
    for feature, _, _, weight in transformed_explanation:
        weight_float = float(weight) if weight is not None else 0.0
        contribution = "Class 1 [Surgery required (Yes)]" if weight_float > 0 else "Class 0 [Surgery not required (No)]"
        feature_contributions[feature] = contribution

    # **4. Feature Importance 바 그래프 생성 (Class별 색상 반영)**
    encoded_importance_image = plot_relative_importance(feature_importance, feature_contributions)

    # **5. HTML 생성**
    with open(file_name, 'w') as f:
        f.write("<html><body><h2>Transformed Explanation</h2>\n")

        # **Prediction Probabilities 그래프**
        f.write("<h3>Prediction Probabilities</h3>\n")
        f.write(f'<img src="data:image/png;base64,{encoded_prob_image}" alt="Prediction Probabilities">\n')

        # **Relative Importance 바 그래프 (새로운 <h3> 태그 추가)**
        f.write("<h3>Relative Importance of Features</h3>\n")
        f.write(f'<img src="data:image/png;base64,{encoded_importance_image}" alt="Relative Importance">\n')

        # **Feature Contributions Table**
        f.write("<h3>Feature Contributions</h3>\n")
        f.write("<table border='1'>\n")
        f.write(
            "<tr><th>Feature</th><th>Original Value</th><th>Transformed Value</th><th>Weight</th><th>Contribution</th><th>Relative Importance</th></tr>\n")

        clinic_features = ['age', 'bmi', 'gender', 'side', 'presence']

        for feature, original_value, transformed_value, weight in transformed_explanation:
            original_value = original_value if original_value is not None else "N/A"
            transformed_value = transformed_value if transformed_value is not None else "N/A"
            weight_str = f"{weight:.6f}" if weight is not None else "N/A"

            # **Weight 부호만 판단하여 기여도 결정**
            try:
                weight_float = float(weight)
                contribution = "Class 1 [Surgery required (Yes)]" if weight_float > 0 else "Class 0 [Surgery not required (No)]"
            except ValueError:
                contribution = "N/A"

            # **Relative Importance 값 찾기 (정확한 매칭)**
            relative_importance = 0.0
            for key in clinic_features:
                if key in feature:  # 정확한 키워드 포함 여부 확인
                    relative_importance = feature_importance.get(key, 0.0)
                    break  # 하나만 매칭되면 중단

            rel_importance_str = f"{relative_importance:.6f}"  # 🔥 소수점 6자리 출력

            f.write(
                f"<tr><td>{feature}</td><td>{original_value}</td><td>{transformed_value}</td><td>{weight_str}</td><td>{contribution}</td><td>{rel_importance_str}</td></tr>\n")

        f.write("</table></body></html>")


def plot_relative_importance(feature_importance, feature_contributions):
    """
    Feature Importance를 바 그래프로 시각화하며, Class 0 / Class 1 기여 여부에 따라 색상을 다르게 적용.
    """
    features, importance = list(feature_importance.keys()), list(feature_importance.values())

    # **각 Feature가 Class 0인지 Class 1에 기여하는지 확인 (정확한 매칭)**
    def get_contribution_label(feature):
        for key in feature_contributions:
            if feature.startswith(key) or key.startswith(feature):
                return feature_contributions[key]
        return "Class 0 [Surgery not required (No)]"  # 기본값: Class 0

    colors = ['blue' if "Class 0" in get_contribution_label(feat) else 'orange' for feat in features]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(features, importance, color=colors)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance (Class 0 = Blue, Class 1 = Orange)", fontsize=12, fontweight='bold')

    # **막대 위에 중요도 값 추가**
    for bar, value, color in zip(bars, importance, colors):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.4f}",
                ha='left', fontsize=9, fontweight='bold', color=color)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    return encoded_image


def compute_relative_importance(transformed_explanation):
    """
    clinic_inputs 내부 feature들의 상대적 중요도를 정규화하여 계산.
    """
    clinic_features = ['age', 'bmi', 'gender', 'side', 'presence']
    feature_weights = {feat: 0 for feat in clinic_features}

    for feature, _, _, weight in transformed_explanation:
        for key in clinic_features:
            if key in feature:
                feature_weights[key] += abs(weight)

    total_weight = sum(feature_weights.values())
    if total_weight > 0:
        feature_importance = {key: feature_weights[key] / total_weight for key in clinic_features}
    else:
        feature_importance = {key: 0 for key in clinic_features}

    return feature_importance


def save_all_lime_results(explanations, age_scaler, bmi_scaler, gender_encoder, side_encoder, presence_encoder,
                          base_dir="lime_results"):
    """
    모든 LIME 결과를 클래스별 디렉토리에 HTML 파일로 저장.

    Args:
        explanations: LIME 결과 리스트 (sample_id, label, explanation 형태).
        base_dir: 결과를 저장할 기본 디렉토리. 기본값은 'lime_results'.
    """
    # 클래스별 디렉토리 생성
    os.makedirs(f"{base_dir}/0", exist_ok=True)
    os.makedirs(f"{base_dir}/1", exist_ok=True)

    # 특정 샘플에 대한 설명 출력
    for sample_id, label, explanation in explanations:
        # 파일 경로 설정
        file_name = f"{base_dir}/{label}/lime_explanation_sample_{sample_id}.html"

        # 모델 예측 확률 가져오기
        prediction_probabilities = explanation.predict_proba

        # 정규화된 데이터에서 원본 값으로 설명 기준을 변환.
        transformed_explanation = explain_with_original_data_and_ranges(explanation,
                                                                        age_scaler=age_scaler,
                                                                        bmi_scaler=bmi_scaler,
                                                                        gender_encoder=gender_encoder,
                                                                        side_encoder=side_encoder,
                                                                        presence_encoder=presence_encoder)

        # HTML 파일로 저장
        try:
            explanation.save_to_file(file_name)
            save_transformed_explanation_html(transformed_explanation, f"{file_name}.html", prediction_probabilities)
            # print(f"Explanation for sample {sample_id} saved to {file_name}")
        except Exception as e:
            print(f"Error saving explanation for sample {sample_id}: {e}")


def predict_fn_for_lime(data, preap_input, prelat_input, combinedModel, batch_size=32):
    # def predict_fn_for_lime(data, preap_input, prelat_input, combinedModel):
    """
    LIME이 호출할 예측 함수. 변형된 테이블 데이터와 고정된 preap_input 및 prelat_input을 사용하여 모델 예측 확률 반환.

    Args:
        data: LIME이 변형한 클리닉 데이터 (numpy 배열, 배치 형태).
        preap_input: 고정된 preap_input (torch.Tensor).
        prelat_input: 고정된 prelat_input (torch.Tensor).
        combinedModel: CombinedModel (PyTorch 모델).

    Returns:
        확률값 배열 (numpy).
    """

    # combinedModel.to(preap_input.device)

    # print("data.shape == ", data.shape) # LIME이 변형한 클리닉 데이터이기 때문에 차원이 높다

    # LIME 변형 데이터를 PyTorch 텐서로 변환
    clinic_input = torch.tensor(data, dtype=torch.float32).to(preap_input.device)
    # 배치로 perturbation을 나누어서 실행
    probs_list = []
    for i in range(0, len(clinic_input), batch_size):
        batch_clinic_input = clinic_input[i:i + batch_size]
        preap_batch = preap_input.expand(batch_clinic_input.size(0), -1, -1, -1)
        prelat_batch = prelat_input.expand(batch_clinic_input.size(0), -1, -1, -1)

        with torch.no_grad():
            logits = combinedModel(preap_batch, prelat_batch, batch_clinic_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.extend(probs)

    return np.array(probs_list)

    # 원본 버젼
    # batch_size = clinic_input.size(0) # LIME이 변형한 클리닉 데이터와 batch를 맞춘다.

    # preap_input = preap_input.expand(batch_size, -1, -1, -1)  # 동일한 값을 배치 크기만큼 확장
    # prelat_input = prelat_input.expand(batch_size, -1, -1, -1)

    # with torch.no_grad():
    #     logits = combinedModel(preap_input, prelat_input, clinic_input)
    #     probs = torch.softmax(logits, dim=1).cpu().numpy()

    # return probs


def explain_instance(testloader, explainer, combinedModel, device='cuda', max_samples=20):
    """
       배치 데이터에서 각 샘플에 대해 LIME 설명을 생성.

       Args:
           testloader: 데이터 로더 (torch.utils.data.DataLoader).
           explainer: LimeTabularExplainer 객체.
           combinedModel: CombinedModel (PyTorch 모델).
           device: GPU 또는 CPU.
       Returns:
           list: 각 샘플에 대한 LIME 설명 객체 리스트.
       """

    combinedModel.eval()
    combinedModel.to(device)
    explanations = []
    samples_processed = 0

    for batch_idx, (ids, preap_inputs, prelat_inputs, clinic_inputs, labels) in tqdm(enumerate(testloader),
                                                                                     total=len(testloader),
                                                                                     desc="Calculating LIME"):
        preap_inputs = preap_inputs.to(device)
        prelat_inputs = prelat_inputs.to(device)
        clinic_inputs = clinic_inputs.to(device)
        labels = labels.to(device)

        for i in range(clinic_inputs.size(0)):

            # 만약 이미 max_samples 개를 넘었다면 조기 종료
            if samples_processed >= max_samples:
                break

            # 현재 샘플 추출
            id = ids[i]
            preap_input = preap_inputs[i].unsqueeze(0)  # (1, C, H, W)
            prelat_input = prelat_inputs[i].unsqueeze(0)  # (1, C, H, W)
            # clinic_input = clinic_inputs[i].unsqueeze(0)
            clinic_input = clinic_inputs[i].cpu().numpy()  # LIME은 numpy 배열 사용
            label = labels[i].item()

            # print("clinic_input == ", clinic_input)

            # LIME 설명 생성
            def lime_predict_fn(data):
                # print("lime_predict_fn data == ", data)
                return predict_fn_for_lime(data, preap_input, prelat_input, combinedModel)

            num_features = clinic_input.shape[0]  # 설명할 클리닉 데이터 feature 개수

            explanation = explainer.explain_instance(
                clinic_input,  # 설명할 클리닉 데이터 (개별 샘플)
                lime_predict_fn,  # 변형된 데이터 예측 함수, 원본 데이터를 변형하는 이유는 모델이 특정 특성(feature)에 얼마나 의존하는지 분석하기 위해서
                num_features=num_features,
                num_samples=num_samples  # perturbation 샘플 개수
            )
            explanations.append((id, label, explanation))
            samples_processed += 1

        # 배치 루프 탈출 조건
        if samples_processed >= max_samples:
            break

    return explanations