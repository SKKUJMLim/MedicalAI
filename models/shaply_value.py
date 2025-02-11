import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def compute_shap_values(model, X, feature_names, age_scaler, bmi_scaler):
    """
    모델과 데이터를 기반으로 SHAP 값을 계산하고, 연속형 변수를 원래 값으로 복원.

    Args:
        model: 훈련된 모델
        X: 예측할 데이터 (정규화된 상태)
        feature_names: 특성 이름 리스트
        age_scaler: Age를 복구할 Scaler
        bmi_scaler: BMI를 복구할 Scaler

    Returns:
        SHAP 값이 포함된 DataFrame
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)  # SHAP 계산

    # 원래 값으로 변환할 변수들 (정규화된 값 → 원래 값)
    transformed_X = X.copy()
    transformed_X[:, 0] = age_scaler.inverse_transform(X[:, [0]]).flatten()  # Age
    transformed_X[:, 1] = bmi_scaler.inverse_transform(X[:, [1]]).flatten()  # BMI

    # SHAP 값과 함께 원래 데이터를 DataFrame으로 정리
    shap_df = pd.DataFrame(transformed_X, columns=feature_names)
    shap_df['SHAP Value'] = np.mean(shap_values.values, axis=0)  # 평균 SHAP 값 추가

    return shap_df


def plot_shap_summary(shap_values, X, feature_names, age_scaler, bmi_scaler):
    """
    SHAP 요약 그래프를 생성하며, 연속형 변수를 원래 값으로 변환.

    Args:
        shap_values: SHAP 분석 결과
        X: 입력 데이터 (정규화된 상태)
        feature_names: 특성 이름 리스트
        age_scaler, bmi_scaler: 연속형 변수 복구용 Scaler

    Returns:
        Base64 인코딩된 그래프 이미지 (HTML 삽입 가능)
    """
    transformed_X = X.copy()
    transformed_X[:, 0] = age_scaler.inverse_transform(X[:, [0]]).flatten()  # Age
    transformed_X[:, 1] = bmi_scaler.inverse_transform(X[:, [1]]).flatten()  # BMI

    shap.summary_plot(shap_values, transformed_X, feature_names=feature_names, show=False)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return encoded_image


def save_shap_results(model, X, feature_names, age_scaler, bmi_scaler, file_name):
    """
    SHAP 분석 결과를 HTML 파일로 저장하며, 연속형 변수(Age, BMI)를 원래 값으로 변환.

    Args:
        model: 훈련된 모델
        X: 예측할 데이터 (정규화된 상태)
        feature_names: 특성 이름 리스트
        age_scaler, bmi_scaler: 연속형 변수 복구용 Scaler
        file_name: 저장할 HTML 파일 이름
    """
    shap_df = compute_shap_values(model, X, feature_names, age_scaler, bmi_scaler)
    encoded_shap_image = plot_shap_summary(shap_df['SHAP Value'].values, X, feature_names, age_scaler, bmi_scaler)

    with open(file_name, "w") as f:
        f.write("<html><body><h2>SHAP Feature Importance</h2>\n")

        # **SHAP 요약 그래프 추가**
        f.write("<h3>SHAP Summary Plot</h3>\n")
        f.write(f'<img src="data:image/png;base64,{encoded_shap_image}" alt="SHAP Summary">\n')

        # **SHAP Feature Contributions Table**
        f.write("<h3>Feature Contributions</h3>\n")
        f.write("<table border='1'>\n")
        f.write("<tr><th>Feature</th><th>Original Value</th><th>SHAP Value</th></tr>\n")

        for _, row in shap_df.iterrows():
            feature_name = row.name
            original_value = row[feature_name]
            shap_value = row['SHAP Value']
            f.write(f"<tr><td>{feature_name}</td><td>{original_value:.2f}</td><td>{shap_value:.6f}</td></tr>\n")

        f.write("</table></body></html>")


### 사용예제
# save_shap_results(trained_model, X_test, ['age', 'bmi', 'gender', 'side', 'presence'], age_scaler, bmi_scaler, "shap_results.html")