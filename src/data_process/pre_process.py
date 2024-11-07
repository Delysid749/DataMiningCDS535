import time
import numpy as np
import pandas as pd

def data_preprocessing(data, is_train=True, categorical_columns=None, numerical_columns=None, one_hot_columns=None):
    # 检查 data 是否为 DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入的数据必须是 pandas DataFrame。")
    dataset_type = "训练集" if is_train else "测试集"
    start_time = time.time()
    print(f"\n=== 开始 {dataset_type} 数据预处理 ===")

    # 替换特殊缺失值
    data.replace([-1, -2, -3, -8], np.nan, inplace=True)

    if is_train and 'happiness' in data.columns:
        print("清洗 'happiness' 列中的缺失值...")
        initial_rows = len(data)
        data.dropna(subset=['happiness'], inplace=True)
        print(f"删除缺失值后，训练集大小从 {initial_rows} 减少到 {len(data)} 行")

    # 处理类别列和数值列
    for col in data.columns:
        if col in categorical_columns:
            data[col] = data[col].fillna(data[col].mode()[0]).astype('category').cat.codes
        elif col in numerical_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].median())

    # 独热编码
    if one_hot_columns:
        data = pd.get_dummies(data, columns=one_hot_columns)

    # 对数变换
    for col in ['income', 'family_income']:
        if col in data.columns:
            data[col] = np.log1p(data[col])

    print(f"{dataset_type} 数据预处理完成，耗时: {time.time() - start_time:.2f} 秒\n")
    return data
