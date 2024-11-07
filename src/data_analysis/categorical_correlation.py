import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif

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

# 输出卡方检验结果并分类输出高、中、低相关性特征及其分数
print("=== 卡方检验结果 ===")
print("根据卡方检验的 Chi2 Score 和 p-value 结果，以下是特征的相关性排序：")
print(chi2_results.sort_values(by="Chi2 Score", ascending=False))

high_significance_features = chi2_results[chi2_results['Significance'] == 'High']
medium_significance_features = chi2_results[chi2_results['Significance'] == 'Medium']
low_significance_features = chi2_results[chi2_results['Significance'] == 'Low']

print("\n高相关性特征 (High):")
print(high_significance_features[['Feature', 'Chi2 Score', 'p-value']])

print("\n中相关性特征 (Medium):")
print(medium_significance_features[['Feature', 'Chi2 Score', 'p-value']])

print("\n低相关性特征 (Low):")
print(low_significance_features[['Feature', 'Chi2 Score', 'p-value']])

# 将低相关性特征作为数组输出
low_correlation_features = low_significance_features['Feature'].tolist()
print("\n低相关性特征数组:", low_correlation_features)

