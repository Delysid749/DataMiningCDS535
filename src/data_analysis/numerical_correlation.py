import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif

# 加载预处理后的数据
file_path = '../../data/processed_train.csv'
data = pd.read_csv(file_path)

# 计算斯皮尔曼相关系数
spearman_income_corr, _ = spearmanr(data['income'], data['happiness'])
spearman_family_income_corr, _ = spearmanr(data['family_income'], data['happiness'])

print("斯皮尔曼相关系数：")
print(f"income 与 happiness 的相关性: {spearman_income_corr:.4f}")
print(f"family_income 与 happiness 的相关性: {spearman_family_income_corr:.4f}")

# 计算方差分析（ANOVA）得分
X = data[['income', 'family_income']]
y = data['happiness']
anova_f_values, anova_p_values = f_classif(X, y)

print("\n方差分析（ANOVA）得分：")
print(f"income 的 F 值: {anova_f_values[0]:.4f}, p 值: {anova_p_values[0]:.4f}")
print(f"family_income 的 F 值: {anova_f_values[1]:.4f}, p 值: {anova_p_values[1]:.4f}")
