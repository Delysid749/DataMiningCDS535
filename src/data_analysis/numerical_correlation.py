
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 加载预处理后的数据
file_path = '../../data/processed_train.csv'
train_data = pd.read_csv(file_path)


# 1. 计算皮尔逊相关系数
correlation_income = train_data['income'].corr(train_data['happiness'])
correlation_family_income = train_data['family_income'].corr(train_data['happiness'])
print(f"'income' 与 'happiness' 的相关性: {correlation_income:.4f}")
print(f"'family_income' 与 'happiness' 的相关性: {correlation_family_income:.4f}")

# 2. 计算不同幸福水平下的收入均值
income_by_happiness = train_data.groupby('happiness')['income'].mean()
family_income_by_happiness = train_data.groupby('happiness')['family_income'].mean()
print("不同幸福水平下的平均收入:\n", income_by_happiness)
print("不同幸福水平下的平均家庭收入:\n", family_income_by_happiness)

# 3. 可视化不同幸福水平下的收入和家庭收入分布
plt.figure(figsize=(12, 5))
sns.boxplot(x='happiness', y='income', data=train_data)
plt.title("Income across different levels of Happiness")
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(x='happiness', y='family_income', data=train_data)
plt.title("Family Income across different levels of Happiness")
plt.show()

# 4. 使用特征重要性分析
# 这里我们使用一个简单的随机森林分类器进行示例
# 首先去除目标列和无关列
X = train_data[['income', 'family_income']]
y = train_data['happiness']

# 将数据分成训练集和测试集（为了确保特征重要性更具有代表性）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 获取特征重要性
feature_importances = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

print("\n特征重要性分析：")
print(feature_importance_df)
