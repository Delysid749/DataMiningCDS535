import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 matplotlib 参数以显示中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据集
file_path = '../../data/happiness_train.csv'
data = pd.read_csv(file_path)

# 描述性统计分析
print("描述性统计分析:")
print(data[['income', 'family_income']].describe())

# 数据分布可视化 - 直方图和核密度图
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['income'], kde=True, bins=30)
plt.title('Income 分布')
plt.xlabel('Income')

plt.subplot(1, 2, 2)
sns.histplot(data['family_income'], kde=True, bins=30)
plt.title('Family Income 分布')
plt.xlabel('Family Income')

plt.tight_layout()
plt.show()

# 缺失值检测
print("\n缺失值检测:")
print(data[['income', 'family_income']].isnull().sum())

# 异常值检测 - 箱线图
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=data['income'])
plt.title('Income 异常值检测')

plt.subplot(1, 2, 2)
sns.boxplot(x=data['family_income'])
plt.title('Family Income 异常值检测')

plt.tight_layout()
plt.show()

# 特征间的相关性分析 - 散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x='income', y='family_income', data=data)
plt.title('Income 与 Family Income 的关系')
plt.xlabel('Income')
plt.ylabel('Family Income')
plt.show()

# 特征与标签关系 - 箱线图和分组均值图
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='happiness', y='income', data=data)
plt.title('Happiness 与 Income 的关系')

plt.subplot(1, 2, 2)
sns.barplot(x='happiness', y='family_income', data=data)
plt.title('Happiness 与 Family Income 的关系')

plt.tight_layout()
plt.show()

# 趋势和模式 - 回归图
plt.figure(figsize=(8, 6))
sns.regplot(x='income', y='happiness', data=data, scatter_kws={'s': 10}, line_kws={'color': 'red'})
plt.title('Income 对 Happiness 的线性趋势')
plt.xlabel('Income')
plt.ylabel('Happiness')
plt.show()
