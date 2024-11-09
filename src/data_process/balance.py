import time
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


def balance_data(train_data, target_column='happiness', id_column='id', model_type='tree'):
    """
    根据模型类型对数据集进行平衡处理，使用 SMOTETomek 或 SMOTE 方法。

    参数:
        train_data (pd.DataFrame): 包含特征和目标变量的数据集。
        target_column (str): 目标变量的列名，用于平衡类别分布。
        id_column (str): ID 列的列名（如果存在），平衡过程中将被忽略。
        model_type (str): 模型类型，可选 'tree'（适用于树模型）或 'linear'（适用于线性模型）。
                          'tree' 模型使用 SMOTETomek 平衡数据，'linear' 模型使用 SMOTE。

    返回:
        tuple: 平衡后的特征 (X_resampled) 和目标变量 (y_resampled)。
    """
    # 记录数据平衡的开始时间
    start_time = time.time()
    print("=== 开始数据平衡 ===")

    # 从原始数据集中分离特征和目标变量
    # 移除 ID 列（如果存在），避免对平衡过程产生影响
    X = train_data.drop(columns=[target_column, id_column], errors='ignore')
    y = train_data[target_column]

    # 打印数据平衡前目标变量的类别分布，帮助了解类别不平衡情况
    print("数据平衡前类别分布:\n", y.value_counts())

    # 根据模型类型选择不同的数据平衡方法
    if model_type == 'tree':
        # 使用适合树模型的 SMOTETomek 方法对数据进行过采样和欠采样
        X_resampled, y_resampled = apply_smote_tomek(X, y)
    elif model_type == 'linear':
        # 使用适合线性模型的 SMOTE 方法对数据进行过采样
        X_resampled, y_resampled = apply_smote(X, y)
    else:
        # 若提供的模型类型不支持，抛出错误提示
        raise ValueError("不支持的模型类型。请使用 'tree' 或 'linear'。")

    # 打印平衡后的类别分布，验证数据是否成功平衡
    print("数据平衡后类别分布:\n", pd.Series(y_resampled).value_counts())

    # 输出数据平衡过程的总耗时
    print(f"数据平衡耗时: {time.time() - start_time:.2f} 秒\n")

    # 返回平衡后的特征和目标变量
    return X_resampled, y_resampled


def apply_smote_tomek(X, y):
    """
    使用 SMOTETomek 进行过采样和欠采样，适用于树模型。

    参数:
        X (pd.DataFrame): 训练集特征数据。
        y (pd.Series): 训练集目标变量。

    返回:
        tuple: 平衡后的特征 (X_resampled) 和目标变量 (y_resampled)。
    """
    print("使用 SMOTETomek 进行数据平衡...")

    # 初始化 SMOTETomek 对象
    # 'sampling_strategy' 设置为 'auto'，自动确定采样比例，以平衡类别
    # 'k_neighbors' 设置为目标类别中最小数量的邻居数，以确保平衡过程不超出数据集容量
    smote_tomek = SMOTETomek(
        sampling_strategy='auto',
        smote=SMOTE(k_neighbors=min(3, y.value_counts().min() - 1))
    )

    # 应用 SMOTETomek 方法对数据进行过采样和欠采样
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return X_resampled, y_resampled


def apply_smote(X, y):
    """
    使用简单 SMOTE 进行过采样，适用于线性模型。

    参数:
        X (pd.DataFrame): 训练集特征数据。
        y (pd.Series): 训练集目标变量。

    返回:
        tuple: 平衡后的特征 (X_resampled) 和目标变量 (y_resampled)。
    """
    print("使用 SMOTE 进行数据平衡...")

    # 初始化 SMOTE 对象
    # 'sampling_strategy' 设置为 'auto'，自动确定采样比例
    # 'k_neighbors' 同样限制为目标类别中最小数量的邻居数
    smote = SMOTE(
        sampling_strategy='auto',
        k_neighbors=min(3, y.value_counts().min() - 1)
    )

    # 应用 SMOTE 方法对数据进行过采样
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
