
import time
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def balance_data(train_data, target_column='happiness', id_column='id', model_type='tree'):
    """
    根据模型类型对数据集进行平衡。

    参数:
    train_data (pd.DataFrame): 包含特征和目标变量的数据集。
    target_column (str): 目标变量的列名。
    id_column (str): ID 列的列名（如果存在）。
    model_type (str): 模型类型，可选 'tree'（适用于 LightGBM、Random Forest、XGBoost、GradientBoostingClassifier）或 'linear'（适用于 Logistic Regression）。

    返回:
    tuple: 平衡后的特征 (X_resampled) 和目标变量 (y_resampled)。
    """
    start_time = time.time()
    print("=== 开始数据平衡 ===")

    X = train_data.drop(columns=[target_column, id_column], errors='ignore')
    y = train_data[target_column]
    print("数据平衡前类别分布:\n", y.value_counts())

    if model_type == 'tree':
        # 对于树模型（包括 LightGBM、Random Forest、XGBoost 和 GradientBoosting），使用 SMOTETomek 进行过采样和欠采样
        smote_tomek = SMOTETomek(sampling_strategy='auto', smote=SMOTE(k_neighbors=min(3, y.value_counts().min() - 1)))
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    elif model_type == 'linear':
        # 对于线性模型（如 Logistic Regression），可以使用简单的 SMOTE
        smote = SMOTE(sampling_strategy='auto', k_neighbors=min(3, y.value_counts().min() - 1))
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        raise ValueError("不支持的模型类型。请使用 'tree' 或 'linear'.")

    print("数据平衡后类别分布:\n", pd.Series(y_resampled).value_counts())
    print(f"数据平衡耗时: {time.time() - start_time:.2f} 秒\n")
    return X_resampled, y_resampled
