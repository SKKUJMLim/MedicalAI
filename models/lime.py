import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import sys
import re


num_samples = 10 # perturbation 샘플 개수

def explain_with_original_data_and_ranges(explanation, age_scaler, bmi_scaler, gender_encoder, side_encoder, presence_encoder):
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
        bmi_scaler.inverse_transform(normalized_instance[:, [1]].astype(float))[0][0]   # BMI
    ]

    gender_value = -1  # 예외처리된 경우를 구분할 수 있도록 기본값 설정 (ex: -1)
    side_value = -1  # 예외처리된 경우를 구분할 수 있도록 기본값 설정 (ex: -1)
    presence_value = -1  # 예외처리된 경우를 구분할 수 있도록 기본값 설정 (ex: -1)

    print("gender_value == ", normalized_instance[:, 2][0])
    print("side_value == ", normalized_instance[:, 3][0])
    print("presence_value == ", normalized_instance[:, 4][0])

    # 범주형 변수(Gender, Side, Presence)는 LabelEncoder를 사용하여 원래 값 복원
    try:
        gender_value = int(float(normalized_instance[:, 2][0]))
        side_value = int(float(normalized_instance[:, 3][0]))
        presence_value = int(float(normalized_instance[:, 4][0]))

        gender_text = gender_encoder.inverse_transform(np.array([gender_value]).astype(int))[0]
        side_text = side_encoder.inverse_transform(np.array([side_value]).astype(int))[0]
        presence_text = presence_encoder.inverse_transform(np.array([presence_value]).astype(int))[0]
    except ValueError as e:
        print(f"Error converting categorical values: {e}")
        gender_text, side_text, presence_text = "Unknown", "Unknown", "Unknown"

    # 문자열에서 숫자만 추출하는 함수
    def extract_float(value):
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        return float(match.group()) if match else None

    # 숫자를 변환하는 함수 (연속형 변수만)
    def transform_number(value, scaler):
        number = extract_float(value)
        if number is not None:
            try:
                return f"{scaler.inverse_transform(np.array([[number]]))[0][0]:.2f}"
            except ValueError:
                return value
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
            transformed_value = transform_number(feature, scaler)

        elif 'bmi' in feature:
            scaler = bmi_scaler
            original_value = original_values[1]  # BMI 원래 값
            transformed_value = transform_number(feature, scaler)

        elif 'gender' in feature:
            original_value = gender_text  # Gender 원래 값
            transformed_value = f"{gender_value} ({gender_text})"

        elif 'side' in feature:
            original_value = side_text  # Side 원래 값
            transformed_value = f"{side_value} ({side_text})"

        elif 'presence' in feature:
            original_value = presence_text  # Presence 원래 값
            transformed_value = f"{presence_value} ({presence_text})"

        else:
            transformed_explanation.append((feature, None, None, weight))
            continue

        # 변환된 feature 추가
        transformed_explanation.append((feature, original_value, transformed_value, weight))

    return transformed_explanation

def save_transformed_explanation_html(transformed_explanation, file_name):
    """
    변환된 설명을 HTML 파일로 저장하며, 컬럼 순서를 `age -> gender -> side -> presence`로 정렬.

    Args:
        transformed_explanation: 변환된 설명 리스트 (feature, 원래값, 변환된값, weight 형태).
        file_name: 저장할 파일 이름.
    """
    # 🎯 **순서대로 정렬**
    sorted_features = ["age", "bmi", "gender", "side", "presence"]  # 원하는 정렬 순서
    sorted_explanation = sorted(
        transformed_explanation,
        key=lambda x: sorted_features.index(x[0].split(" ")[0]) if x[0].split(" ")[0] in sorted_features else len(sorted_features)
    )

    with open(file_name, 'w') as f:
        f.write("<html><body><h2>Transformed Explanation</h2>\n")
        f.write("<table border='1'>\n")
        f.write("<tr><th>Feature</th><th>Original Value</th><th>Transformed Value</th><th>Weight</th></tr>\n")

        for feature, original_value, transformed_value, weight in sorted_explanation:
            original_value = original_value if original_value is not None else "N/A"
            transformed_value = transformed_value if transformed_value is not None else "N/A"
            weight = f"{weight:.4f}" if weight is not None else "N/A"

            f.write(f"<tr><td>{feature}</td><td>{original_value}</td><td>{transformed_value}</td><td>{weight}</td></tr>\n")

        f.write("</table></body></html>")


def save_all_lime_results(explanations, age_scaler, bmi_scaler, gender_encoder, side_encoder, presence_encoder, base_dir="lime_results"):
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
            save_transformed_explanation_html(transformed_explanation, f"{file_name}.html")
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

    combinedModel.to(preap_input.device)

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

    #원본 버젼
    # batch_size = clinic_input.size(0) # LIME이 변형한 클리닉 데이터와 batch를 맞춘다.

    # preap_input = preap_input.expand(batch_size, -1, -1, -1)  # 동일한 값을 배치 크기만큼 확장
    # prelat_input = prelat_input.expand(batch_size, -1, -1, -1)

    # with torch.no_grad():
    #     logits = combinedModel(preap_input, prelat_input, clinic_input)
    #     probs = torch.softmax(logits, dim=1).cpu().numpy()

    # return probs


def explain_instance(testloader, explainer, combinedModel, device='cuda', max_samples= 20):
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