import pandas as pd

# 加载Excel文件中的数据
file_path = '../../data/happiness_index.xlsx'
df = pd.read_excel(file_path)

# 设置显示选项，确保控制台输出完整内容
pd.set_option('display.max_colwidth', None)  # 确保列内容不被截断

# 定义您需要的六个属性名
target_attributes = ['class', 'depression', 'equity', 'health', 'family_status', 'health_problem']

# 从数据中筛选出这些属性名及其对应的取值含义
filtered_df = df[df['变量名'].isin(target_attributes)]

# 将筛选后的结果转换为单行字符串输出
print(filtered_df[['变量名', '问题', '取值含义']].to_string(index=False, header=False))

# 恢复默认设置（可选）
pd.reset_option('display.max_colwidth')
