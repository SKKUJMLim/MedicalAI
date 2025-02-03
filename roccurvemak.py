import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np

# 1. CSV 파일 불러오기 (예시: roc_data_onlyimage.csv, roc_data_combined.csv)
df_only = pd.read_csv(r"C:\융합과제 의대\roc_data_only_01_19.csv")
df_combined = pd.read_csv(r"C:\융합과제 의대\roc_data_new0203.csv.csv")

for df in [df_only, df_combined]:
    # 만약 열 이름이 조금 다르면 컬럼명을 맞춰주세요
    df['False Positive Rate'] = pd.to_numeric(df['False Positive Rate'], errors='coerce')
    df['True Positive Rate'] = pd.to_numeric(df['True Positive Rate'], errors='coerce')
    # 변환 안 되는 값은 NaN이 되므로 dropna
    df.dropna(subset=['False Positive Rate', 'True Positive Rate'], inplace=True)
    # inf 제거
    df = df[~np.isinf(df['False Positive Rate'])]
    df = df[~np.isinf(df['True Positive Rate'])]

# 3) FPR, TPR 추출
fpr_only = df_only['False Positive Rate'].values
tpr_only = df_only['True Positive Rate'].values

fpr_combined = df_combined['False Positive Rate'].values
tpr_combined = df_combined['True Positive Rate'].values

# 4. AUC 계산 (각 ROC 곡선에 대한 AUC)
auc_only = auc(fpr_only, tpr_only)
auc_combined = auc(fpr_combined, tpr_combined)

# 두 곡선 사이를 강조하는 채우기 영역 추가
# plt.fill_between(fpr_only, tpr_only, tpr_combined, color='gray', alpha=0.2, label='Difference region')

# 5. ROC 곡선 그리기
plt.figure(figsize=(8, 6))
plt.plot(fpr_combined, tpr_combined, color='blue', lw=2, 
         label=f'Image + Clinical data model (AUC = {auc_combined:.4f})')
plt.plot(fpr_only, tpr_only, color='orange', lw=2, 
         label=f'Image only model (AUC = {auc_only:.4f})')



# 대각선 기준선 (무작위 분류선) 추가
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')


# 촘촘한 눈금 설정
plt.xticks(np.arange(0.0, 1.01, 0.1))
plt.yticks(np.arange(0.0, 1.01, 0.1))

# 축 및 제목, 범례 설정
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve Comparison', fontsize=14,fontweight='bold')
plt.legend(loc="lower right")

# 그림 저장 및 표시
plt.savefig('combined_roc_curve.png', bbox_inches='tight')
plt.show()
