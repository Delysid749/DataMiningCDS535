import time
import numpy as np
import pandas as pd


def data_preprocessing(data, is_train=True, categorical_columns=None, numerical_columns=None, one_hot_columns=None):
    """数据预处理函数，处理缺失值、类别编码、数值填充和独热编码等操作。

    参数:
        data (pd.DataFrame): 输入的数据集
        is_train (bool): 是否为训练集，默认为 True
        categorical_columns (list): 类别特征列列表
        numerical_columns (list): 数值特征列列表
        one_hot_columns (list): 需要独热编码的特征列列表

    返回:
        pd.DataFrame: 预处理后的数据集
    """
    # 初始检查：确认输入数据格式
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入的数据必须是 pandas DataFrame。")

    dataset_type = "训练集" if is_train else "测试集"
    print(f"\n=== 开始 {dataset_type} 数据预处理 ===")
    start_time = time.time()

    # 处理特殊缺失值标记
    replace_missing_values(data)

    # 如果是训练集，清理目标列 'happiness' 中的缺失值
    if is_train and 'happiness' in data.columns:
        clean_target_column(data, 'happiness')

    # 处理类别列和数值列的缺失值
    encode_categorical_columns(data, categorical_columns)
    fill_numerical_columns(data, numerical_columns)

    # 对指定列进行独热编码
    if one_hot_columns:
        data = pd.get_dummies(data, columns=one_hot_columns)

    # 对收入类列进行对数变换
    apply_log_transformation(data, ['income', 'family_income'])

    print(f"{dataset_type} 数据预处理完成，耗时: {time.time() - start_time:.2f} 秒\n")
    return data


def replace_missing_values(data):
    """替换特殊缺失值标记为 NaN。"""
    data.replace([-1, -2, -3, -8], np.nan, inplace=True)


def clean_target_column(data, target_column):
    """清洗目标列中的缺失值，仅适用于训练集。"""
    print(f"清洗 '{target_column}' 列中的缺失值...")
    initial_rows = len(data)
    data.dropna(subset=[target_column], inplace=True)
    print(f"删除缺失值后，{target_column} 列从 {initial_rows} 行减少到 {len(data)} 行")


def encode_categorical_columns(data, categorical_columns):
    """编码类别特征列，填充缺失值为众数，并转化为分类编码。"""
    if categorical_columns:
        for col in categorical_columns:
            data[col] = data[col].fillna(data[col].mode()[0]).astype('category').cat.codes


def fill_numerical_columns(data, numerical_columns):
    """填充数值特征列的缺失值为中位数。"""
    if numerical_columns:
        for col in numerical_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].median())


def apply_log_transformation(data, columns):
    """对指定列进行对数变换，平滑数据分布。"""
    for col in columns:
        if col in data.columns:
            data[col] = np.log1p(data[col])  # 使用 log1p 以避免 log(0) 错误
