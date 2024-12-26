import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm




num_samples=5000
 # perturbation 샘플 개수

def save_all_lime_results(explanations):

    os.makedirs(f'lime_results/0', exist_ok=True)
    os.makedirs(f'lime_results/1', exist_ok=True)

    # 특정 샘플에 대한 설명 출력
    for sample_id, label, explanation in explanations:
        # print(f"Sample ID: {sample_id}, Label: {label}")
        file_name = f"lime_results/{label}/lime_explanation_sample_{sample_id}.html"
        explanation.show_in_notebook()
        explanation.save_to_file(file_name)


def predict_fn_for_lime(data, preap_input, prelat_input, combinedModel, batch_size = 16):
#def predict_fn_for_lime(data, preap_input, prelat_input, combinedModel):
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
    #배치로 perturbation을 나누어서 실행
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



def explain_instance(testloader, explainer, combinedModel, device='cuda'):
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


    # combinedModel.eval()
    # combinedModel.to(device)
    explanations = []

    for batch_idx, (ids, preap_inputs, prelat_inputs, clinic_inputs, labels) in tqdm(enumerate(testloader), total=len(testloader), desc="Calculating LIME"):

        preap_inputs = preap_inputs.to(device)
        prelat_inputs = prelat_inputs.to(device)
        clinic_inputs = clinic_inputs.to(device)
        labels = labels.to(device)

        for i in range(clinic_inputs.size(0)):
            # 현재 샘플 추출
            id = ids[i]
            preap_input = preap_inputs[i].unsqueeze(0)   # (1, C, H, W)
            prelat_input = prelat_inputs[i].unsqueeze(0) # (1, C, H, W)
            # clinic_input = clinic_inputs[i].unsqueeze(0)
            clinic_input = clinic_inputs[i].cpu().numpy()  # LIME은 numpy 배열 사용
            label = labels[i].item()

            # print("clinic_input == ", clinic_input)

            # LIME 설명 생성
            def lime_predict_fn(data):
                # print("lime_predict_fn data == ", data)
                return predict_fn_for_lime(data, preap_input, prelat_input, combinedModel)

            num_features = clinic_input.shape[0] # 설명할 클리닉 데이터 feature 개수 (age, bmi)

            explanation = explainer.explain_instance(
                clinic_input,     # 설명할 클리닉 데이터 (개별 샘플)
                lime_predict_fn,  # 변형된 데이터 예측 함수, 원본 데이터를 변형하는 이유는 모델이 특정 특성(feature)에 얼마나 의존하는지 분석하기 위해서
                num_features=num_features,
                num_samples=num_samples # perturbation 샘플 개수
            )

            explanations.append((id, label, explanation))

    return explanations