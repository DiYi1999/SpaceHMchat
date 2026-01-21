"""
读取/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/results/confusion_matrix.csv文件，计算各分类指标

"""
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef, roc_auc_score, cohen_kappa_score



con_matrix_path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/results/confusion_matrix.csv'
con_matrix = pd.read_csv(con_matrix_path, index_col=0)
# 将其内元素转为整数
con_matrix = con_matrix.astype(int)

# 当初计算混淆矩阵时：
# # 计算混淆矩阵，第i行第j的元素表示第i类样本被预测为第j类的数量
# confusion_matrix = np.zeros((len(fault_dict), len(fault_dict)))
# for true_label, pred_label in zip(label_list, result_list):
#     confusion_matrix[true_label-1, pred_label-1] += 1
#     # print(confusion_matrix)

y_true = []
y_pred = []
# labels = con_matrix.columns.tolist()
labels = [i for i in range(0, len(con_matrix))]
print("labels:", labels)
for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):
        count = con_matrix.iloc[i, j]
        y_true.extend([true_label] * count)
        y_pred.extend([pred_label] * count)

# 计算各分类指标
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division='warn')
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division='warn')
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division='warn')

precision_micro = precision_score(y_true, y_pred, average='micro', zero_division='warn')
recall_micro = recall_score(y_true, y_pred, average='micro', zero_division='warn')
f1_micro = f1_score(y_true, y_pred, average='micro', zero_division='warn')

roc_auc_ovr = None  # 需要概率值，无法计算
roc_auc_ovo = None  # 需要概率值，无法计算

# Cohen’s Kappa (κ 系数) 是一个 一致性指标，最早用来衡量两个标注者（或者一个模型 vs 真实标签）的分类一致性，它比单纯的准确率更严格，因为它考虑了 随机一致的概率。
# Accuracy：只看预测对了多少，不管是不是因为数据偏分布（比如 90% 样本都是类 0，模型全预测 0 也能有 90% 准确率）。
# Kappa：会扣除掉这种“运气好 / 类别分布不均”造成的假一致。
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
kappa = cohen_kappa_score(y_true, y_pred)

# Compute the Matthews correlation coefficient (MCC).
# The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient. [source: Wikipedia]
# Matthews Correlation Coefficient (MCC) 和 Cohen’s Kappa 有点像，都是为了避免准确率在类别不平衡时的“虚高”，但它更偏向“相关性”的视角。
# Kappa → “比随机好多少？”          MCC → “预测和真实之间的线性相关性有多强？”
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Precision (Micro): {precision_micro:.4f}")
print(f"Recall (Micro): {recall_micro:.4f}")
print(f"F1 Score (Micro): {f1_micro:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"ROC AUC (OvR): {roc_auc_ovr}")
print(f"ROC AUC (OvO): {roc_auc_ovo}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division='warn'))




