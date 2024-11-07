import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
X = data[categorical_columns]

# 1. 卡方检验
chi2_scores, p_values = chi2(X, y)
chi2_results = pd.DataFrame({'Feature': categorical_columns, 'Chi2 Score': chi2_scores, 'p-value': p_values})
chi2_results['Significance'] = pd.cut(
    chi2_results['p-value'],
    bins=[0, 0.01, 0.05, 1],
    labels=['High', 'Medium', 'Low']
)

# 输出卡方检验结果
print("=== 卡方检验结果 ===")
print(chi2_results.sort_values(by="Chi2 Score", ascending=False))

# 2. 互信息
mutual_info_scores = mutual_info_classif(X, y)
mutual_info_results = pd.DataFrame({'Feature': categorical_columns, 'Mutual Information Score': mutual_info_scores})
mutual_info_results['Significance'] = pd.cut(
    mutual_info_results['Mutual Information Score'],
    bins=[0, 0.01, 0.02, 1],
    labels=['Low', 'Medium', 'High']
)

# 输出互信息结果
print("\n=== 互信息得分 ===")
print(mutual_info_results.sort_values(by="Mutual Information Score", ascending=False))

# 3. 使用随机森林计算特征重要性
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
rf_importance = rf_model.feature_importances_

# 使用梯度提升计算特征重要性
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X, y)
gb_importance = gb_model.feature_importances_

# 将特征重要性结果加入到 DataFrame 中
importance_results = pd.DataFrame({
    'Feature': categorical_columns,
    'RandomForest Importance': rf_importance,
    'GradientBoosting Importance': gb_importance
})

importance_results['RF Significance'] = pd.cut(
    importance_results['RandomForest Importance'],
    bins=[0, 0.01, 0.02, 1],
    labels=['Low', 'Medium', 'High']
)

importance_results['GB Significance'] = pd.cut(
    importance_results['GradientBoosting Importance'],
    bins=[0, 0.01, 0.02, 1],
    labels=['Low', 'Medium', 'High']
)

# 输出特征重要性结果
print("\n=== 随机森林和梯度提升特征重要性 ===")
print(importance_results.sort_values(by="RandomForest Importance", ascending=False))

# 4. 综合分析
# 找出在卡方检验、互信息、随机森林和梯度提升中都显示出高相关性的特征
high_significance_features = chi2_results[chi2_results['Significance'] == 'High']['Feature']
high_info_features = mutual_info_results[mutual_info_results['Significance'] == 'High']['Feature']
high_rf_features = importance_results[importance_results['RF Significance'] == 'High']['Feature']
high_gb_features = importance_results[importance_results['GB Significance'] == 'High']['Feature']

# 获取同时在卡方、互信息、随机森林和梯度提升中均为高的特征
common_high_features = set(high_significance_features) & set(high_info_features) & set(high_rf_features) & set(high_gb_features)

# 输出综合分析结果
print("\n=== 综合分析 ===")
if common_high_features:
    print("根据卡方检验、互信息得分、随机森林和梯度提升特征重要性，以下特征同时显示出较高的相关性，建议优先考虑：")
    print(", ".join(common_high_features))
else:
    print("没有高度相关的特征满足卡方检验、互信息得分、随机森林和梯度提升特征重要性的多重标准。")

# 5. 特征相关性总结
print("\n=== 特征相关性总结 ===")
print("高度相关特征（卡方、互信息、随机森林和梯度提升均较高）：", common_high_features)

# 低相关性特征
low_significance_features = chi2_results[chi2_results['Significance'] == 'Low']['Feature']
low_info_features = mutual_info_results[mutual_info_results['Significance'] == 'Low']['Feature']
low_rf_features = importance_results[importance_results['RF Significance'] == 'Low']['Feature']
low_gb_features = importance_results[importance_results['GB Significance'] == 'Low']['Feature']
common_low_features = set(low_significance_features) & set(low_info_features) & set(low_rf_features) & set(low_gb_features)

print("低相关特征（卡方、互信息、随机森林和梯度提升均较低）：", common_low_features)
