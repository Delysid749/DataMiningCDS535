import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import logging
import seaborn as sns

# 配置日志输出路径
def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

setup_logger('../logs/feature_selection_log.txt')

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

# 3. 归一化 Chi2 和 Mutual Info Scores
scaler = MinMaxScaler()
chi2_results['Chi2 Score Normalized'] = scaler.fit_transform(chi2_results[['Chi2 Score']])
chi2_results['Mutual Info Score Normalized'] = scaler.fit_transform(chi2_results[['Mutual Info Score']])

# 自适应加权组合得分计算
def adaptive_combined_score(row):
    if row['p-value'] < 0.01:
        weight_chi2, weight_mutual_info = 0.8, 0.2
    elif row['p-value'] < 0.05:
        weight_chi2, weight_mutual_info = 0.6, 0.4
    else:
        weight_chi2, weight_mutual_info = 0.4, 0.6
    score = weight_chi2 * row['Chi2 Score Normalized'] + weight_mutual_info * row['Mutual Info Score Normalized']
    return score

# 4. 计算自适应加权组合得分并按组合得分排序
chi2_results['Adaptive Combined Score'] = chi2_results.apply(adaptive_combined_score, axis=1)
chi2_results.sort_values(by="Adaptive Combined Score", ascending=False, inplace=True)

# 控制台输出
print("=== 卡方检验和互信息得分结果 ===")
print("根据组合得分排序的特征相关性：")
print(chi2_results[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Adaptive Combined Score', 'p-value']])

# 记录组合得分结果到日志
logging.info("=== 自适应加权组合得分结果 ===")
logging.info("\n" + chi2_results[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Adaptive Combined Score', 'p-value']].to_string(index=False))

# 可视化组合得分结果并添加标签
plt.figure(figsize=(16, 10))
bars = plt.barh(chi2_results['Feature'], chi2_results['Adaptive Combined Score'], color='#1f77b4', edgecolor='black', height=0.6)
plt.xlabel("Adaptive Combined Score", fontsize=14, labelpad=15)
plt.ylabel("Features", fontsize=14, labelpad=15)
plt.title("Feature Importance by Adaptive Combined Score (Chi2 & Mutual Information)", fontsize=14, pad=20)
plt.suptitle("Adaptive Combined Score", fontsize=18, weight='bold', color='darkblue', y=1.02)
plt.xlim(0, 1.1)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 添加每个条形的数值标签
for bar, score, feature in zip(bars, chi2_results['Adaptive Combined Score'], chi2_results['Feature']):
    plt.text(score + 0.02, bar.get_y() + bar.get_height()/2, f'{feature}: {score:.2f}',
             va='center', ha='left', fontsize=10, color='black')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 分类输出高、中、低相关性特征
high_significance_features = chi2_results[chi2_results['Significance'] == 'High']
medium_significance_features = chi2_results[chi2_results['Significance'] == 'Medium']
low_significance_features = chi2_results[chi2_results['Significance'] == 'Low']

print("\n高相关性特征 (High):")
print(high_significance_features[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Adaptive Combined Score', 'p-value']])

print("\n中相关性特征 (Medium):")
print(medium_significance_features[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Adaptive Combined Score', 'p-value']])

print("\n低相关性特征 (Low):")
print(low_significance_features[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Adaptive Combined Score', 'p-value']])

# 剔除高 p-value 的低相关性特征
threshold = 0.05
low_correlation_features = chi2_results[(chi2_results['Significance'] == 'Low') & (chi2_results['p-value'] > threshold)]['Feature'].tolist()
print("\n剔除后的低相关性特征数组:", low_correlation_features)

# 记录低相关性特征数组到日志
logging.info(f"剔除的低相关性特征 (p-value > {threshold}): {low_correlation_features}")
logging.info("=== 特征选择过程完成 ===")

# 仅保留前10个重要特征以提高可读性
top_features = chi2_results[['Feature', 'Chi2 Score', 'Mutual Info Score', 'Adaptive Combined Score']].head(10)
top_features.set_index('Feature', inplace=True)

# 准备热力图数据
heatmap_data = top_features.T  # 转置，使得特征名称在列上

# 绘制热力图
plt.figure(figsize=(16, 8))  # 调整图表大小以便特征名称更清晰
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5, cbar_kws={'label': 'Score'})

# 添加标题和轴标签
plt.title("Top Feature Scores Heatmap (Chi2, Mutual Information, Combined)", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=12)  # 显示特征名称，旋转并调整字体大小
plt.yticks([0.5, 1.5, 2.5], ["Chi2 Score", "Mutual Information Score", "Adaptive Combined Score"], rotation=0, fontsize=12)  # 标注每行的分数类型

plt.tight_layout()  # 调整布局以避免标签重叠
plt.show()

