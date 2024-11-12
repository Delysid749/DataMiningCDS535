import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif
import matplotlib.pyplot as plt

# 加载预处理后的数据
file_path = '../../data/processed_train.csv'
data = pd.read_csv(file_path)

# 定义已经编码的分类特征
categorical_columns = [
    'survey_type', 'province', 'gender', 'nationality', 'religion', 'religion_freq',
    'edu', 'political', 'health', 'health_problem', 'depression', 'hukou',
    'socialize', 'relax', 'learn', 'equity', 'class', 'work_exper',
    'work_status', 'work_type', 'work_manage', 'family_status', 'car', 'marital'
]

# 定义标签
y = data['happiness']

# 1. 卡方检验
chi2_scores, p_values = chi2(data[categorical_columns], y)
chi2_results = pd.DataFrame({'Feature': categorical_columns, 'Chi2 Score': chi2_scores, 'p-value': p_values})
chi2_results['Significance'] = pd.cut(
    chi2_results['p-value'],
    bins=[0, 0.01, 0.05, 1],
    labels=['High', 'Medium', 'Low']
)

# 2. 互信息得分
mutual_info_scores = mutual_info_classif(data[categorical_columns], y, discrete_features=True)
chi2_results['Mutual Info Score'] = mutual_info_scores

# 3. 计算组合得分（Chi2 Score 和 Mutual Info Score 的加权平均）
chi2_results['Combined Score'] = (0.7 * chi2_results['Chi2 Score']) + (0.3 * chi2_results['Mutual Info Score'])

# 4. 输出卡方检验和互信息得分结果，并按组合得分排序
chi2_results.sort_values(by="Combined Score", ascending=False, inplace=True)

print("=== 卡方检验和互信息得分结果 ===")
print("根据组合得分排序的特征相关性：")
print(chi2_results[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Combined Score', 'p-value']])

# 5. 可视化组合得分结果
plt.figure(figsize=(10, 8))
plt.barh(chi2_results['Feature'], chi2_results['Combined Score'], color='skyblue')
plt.xlabel("Combined Score")
plt.title("Feature Importance by Combined Score (Chi2 & Mutual Information)")
plt.gca().invert_yaxis()  # 反转 y 轴使得高分特征在上方
plt.show()

# 分类输出高、中、低相关性特征
high_significance_features = chi2_results[chi2_results['Significance'] == 'High']
medium_significance_features = chi2_results[chi2_results['Significance'] == 'Medium']
low_significance_features = chi2_results[chi2_results['Significance'] == 'Low']

print("\n高相关性特征 (High):")
print(high_significance_features[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Combined Score', 'p-value']])

print("\n中相关性特征 (Medium):")
print(medium_significance_features[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Combined Score', 'p-value']])

print("\n低相关性特征 (Low):")
print(low_significance_features[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Combined Score', 'p-value']])

# 6. 剔除高 p-value 的低相关性特征，设定一个更严格的 p-value 阈值
threshold = 0.05
low_correlation_features = chi2_results[(chi2_results['Significance'] == 'Low') & (chi2_results['p-value'] > threshold)]['Feature'].tolist()
print("\n剔除后的低相关性特征数组:", low_correlation_features)
