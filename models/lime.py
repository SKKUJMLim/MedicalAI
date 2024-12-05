import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from lime.lime_tabular import LimeTabularExplainer

def predict_fn_for_lime(data, preap_input, prelat_input, combinedModel):

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

    with torch.no_grad():
        logits = combinedModel(preap_input, prelat_input, data)
        probs = torch.softmax(logits, dim=1).numpy()

    return probs



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


    combinedModel.eval()
    combinedModel.to(device)
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
            clinic_input = clinic_inputs[i].unsqueeze(0)
            # clinic_input = clinic_inputs[i].cpu().numpy()  # LIME은 numpy 배열 사용
            label = labels[i].item()

            # LIME 설명 생성
            def lime_predict_fn(data):
                return predict_fn_for_lime(preap_input, prelat_input, clinic_input, label, combinedModel)

            num_features = clinic_input.shape[0] # 설명할 클리닉 데이터 개수, print 해보자

            explanation = explainer.explain_instance(
                clinic_input,     # 설명할 클리닉 데이터 (개별 샘플)
                lime_predict_fn,  # 변형된 데이터 예측 함수, 원본 데이터를 변형하는 이유는 모델이 특정 특성(feature)에 얼마나 의존하는지 분석하기 위해서
                num_features
            )

            explanations.append((id, label, explanation))

    return explanations




