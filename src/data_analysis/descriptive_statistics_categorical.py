import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 matplotlib 参数以显示中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据集
file_path = '../../data/happiness_train.csv'
data = pd.read_csv(file_path)

# 替换缺失值
data.replace([-1, -2, -3, -8], pd.NA, inplace=True)

# 分类特征属性列表
categorical_columns = ['survey_type', 'province', 'gender', 'nationality', 'religion',
                       'religion_freq', 'edu', 'political', 'health', 'health_problem',
                       'hukou', 'socialize', 'relax', 'learn', 'equity', 'class',
                       'work_exper', 'work_status', 'work_type', 'work_manage',
                       'family_status', 'car', 'marital']

# 重新分析分类特征
for col in categorical_columns:
    # 查看每个分类特征的值分布
    distribution = data[col].value_counts().sort_index()

    # 绘制分布图
    plt.figure(figsize=(12, 6))
    sns.barplot(x=distribution.index, y=distribution.values)
    plt.title(f'{col} 的值分布')
    plt.xlabel(f'{col} 编码')
    plt.ylabel('频率')
    plt.xticks(rotation=45)
    plt.show()

    # 打印值分布及解释（从解释文档中补充此处）
    print(f"\n{col} 的值分布:")
    print(distribution)
    print(f"{col} 的编码含义请参阅属性解释文档以获取详细信息。\n")
