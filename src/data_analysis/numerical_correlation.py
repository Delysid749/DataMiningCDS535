import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载预处理后的数据
file_path = '../../data/processed_train.csv'
train_data = pd.read_csv(file_path)

# 1. 计算皮尔逊相关系数
correlation_income = train_data['income'].corr(train_data['happiness'])
correlation_family_income = train_data['family_income'].corr(train_data['happiness'])
print(f"'income' 与 'happiness' 的相关性: {correlation_income:.4f}")
print(f"'family_income' 与 'happiness' 的相关性: {correlation_family_income:.4f}")

# 2. 计算不同幸福水平下的收入和家庭收入均值
income_by_happiness = train_data.groupby('happiness')['income'].mean()
family_income_by_happiness = train_data.groupby('happiness')['family_income'].mean()
print("\n不同幸福水平下的平均收入:\n", income_by_happiness)
print("\n不同幸福水平下的平均家庭收入:\n", family_income_by_happiness)

# 3. 可视化不同幸福水平下的收入和家庭收入分布
plt.figure(figsize=(12, 5))
sns.boxplot(x='happiness', y='income', data=train_data)
plt.title("Income Distribution across Different Levels of Happiness")
plt.xlabel("Happiness Level")
plt.ylabel("Income")
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(x='happiness', y='family_income', data=train_data)
plt.title("Family Income Distribution across Different Levels of Happiness")
plt.xlabel("Happiness Level")
plt.ylabel("Family Income")
plt.show()
