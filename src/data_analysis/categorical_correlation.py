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
print("卡方检验结果：")
print(chi2_results.sort_values(by="Chi2 Score", ascending=False))

# 2. 互信息
mutual_info_scores = mutual_info_classif(data[categorical_columns], y)
mutual_info_results = pd.DataFrame({'Feature': categorical_columns, 'Mutual Information Score': mutual_info_scores})
print("\n互信息得分：")
print(mutual_info_results.sort_values(by="Mutual Information Score", ascending=False))
