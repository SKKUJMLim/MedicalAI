import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import sys
import re


num_samples = 5000
# perturbation 샘플 개수

def explain_with_original_data_and_ranges(explanation, age_scaler, bmi_scaler):
    """
    정규화된 데이터로 모델 예측, 원본 값으로 설명 기준을 변환.

    Args:
        explanation: 정규화된 데이터에 대한 LIME 결과
        age_scaler: 정규화에 사용한 Scaler 객체 (MinMaxScaler 또는 StandardScaler).
        bmi_scaler: 정규화에 사용한 Scaler 객체 (MinMaxScaler 또는 StandardScaler).

    Returns:
        변환된 LIME 설명 리스트 (feature, weight 형태).
    """
    import re

    # 문자열에서 숫자만 추출하는 함수
    def extract_float(value):
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        return float(match.group()) if match else None

    # 숫자를 변환하는 함수
    def transform_number(value, scaler):
        number = extract_float(value)
        if number is not None:
            return f"{scaler.inverse_transform([[number]])[0][0]:.2f}"
        return value

    # 변환된 설명 리스트
    transformed_explanation = []

    # 설명 수정: 정규화된 기준을 원본 값으로 변환
    for feature, weight in explanation.as_list():
        if 'age' in feature:
            scaler = age_scaler
        elif 'bmi' in feature:
            scaler = bmi_scaler
        else:
            print(f"LIME inverse_transform Error for feature: {feature}")
            transformed_explanation.append((feature, weight))
            continue

        # feature 내 숫자만 변환
        modified_feature = re.sub(
            r"[-+]?\d*\.\d+|\d+",  # 숫자 패턴
            lambda x: transform_number(x.group(), scaler),  # 숫자 변환
            feature
        )

        # 변환된 feature와 weight 추가
        transformed_explanation.append((modified_feature, weight))

    return transformed_explanation

def save_transformed_explanation_html(transformed_explanation, file_name):
    """
    변환된 설명을 HTML 파일로 저장.

    Args:
        transformed_explanation: 변환된 설명 리스트 (feature, weight 형태).
        file_name: 저장할 파일 이름.
    """
    with open(file_name, 'w') as f:
        f.write("<html><body><h2>Transformed Explanation</h2><ul>\n")
        for feature, weight in transformed_explanation:
            f.write(f"<li>{feature}: {weight:.4f}</li>\n")
        f.write("</ul></body></html>")
    # print(f"Transformed explanation saved to {file_name}")


def save_all_lime_results_with_conversion(explanations, age_scaler, bmi_scaler, base_dir="lime_results"):
    """
    모든 LIME 결과를 클래스별 디렉토리에 HTML 파일로 저장.
    정규화된 데이터를 원본 값으로 변환하여 저장.

    Args:
        explanations: LIME 결과 리스트 (sample_id, label, explanation 형태).
        age_scaler: 정규화에 사용한 Scaler 객체 (MinMaxScaler 또는 StandardScaler).
        bmi_scaler: 정규화에 사용한 Scaler 객체 (MinMaxScaler 또는 StandardScaler).
        base_dir: 결과를 저장할 기본 디렉토리.
    """
    # 클래스별 디렉토리 생성
    os.makedirs(f"{base_dir}/0", exist_ok=True)
    os.makedirs(f"{base_dir}/1", exist_ok=True)

    # 각 샘플에 대해 LIME 결과 저장
    for sample_id, label, explanation in explanations:
        # 정규화된 데이터에서 원본 값으로 설명 기준을 변환
        transformed_explanation = explain_with_original_data_and_ranges(explanation, age_scaler, bmi_scaler)

        # 파일 경로 설정
        file_name = f"{base_dir}/{label}/lime_explanation_sample_{sample_id}.html"

        # HTML 파일로 저장
        try:
            explanation.save_to_file(file_name)
            transformed_explanation.save_to_file(file_name)
            print(f"Explanation for sample {sample_id} saved to {file_name}")
        except Exception as e:
            print(f"Error saving explanation for sample {sample_id}: {e}")


def save_all_lime_results(explanations, age_scaler, bmi_scaler, base_dir="lime_results"):
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

        # print("before===")
        # print(explanation.local_exp)

        # 정규화된 데이터에서 원본 값으로 설명 기준을 변환.
        transformed_explanation = explain_with_original_data_and_ranges(explanation, age_scaler, bmi_scaler)
        #
        # print("after===")
        # print(transformed_explanation)

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

            num_features = clinic_input.shape[0]  # 설명할 클리닉 데이터 feature 개수 (age, bmi)

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