import pandas as pd
import numpy as np
import psutil
import time  # 导入时间模块
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor  # 导入多线程模块


# 内存使用监控
def print_memory_usage(stage=""):
    """
    打印当前 Python 进程的内存使用情况。

    参数:
    stage (str): 当前代码运行阶段的标识。
    """
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 ** 2  # 将字节转换为 MB
    print(f"{stage} - 当前内存使用: {mem:.2f} MB")


# 数据预处理函数
def data_preprocessing(input_file, is_train=True, categorical_columns=None, numerical_columns=None):
    """
    读取数据并进行预处理，包括缺失值处理和数据类型转换。

    参数:
    input_file (str): 数据文件的路径。
    is_train (bool): 是否为训练数据集。
    categorical_columns (list): 分类特征列名列表。
    numerical_columns (list): 数值特征列名列表。

    返回:
    pd.DataFrame: 预处理后的数据集。
    """
    start_time = time.time()  # 开始计时
    print("开始数据预处理...")

    dtype_spec = {column: 'category' for column in categorical_columns}
    data = pd.read_csv(input_file, dtype=dtype_spec)

    # 将特殊值视为缺失值
    special_values = [-1, -2, -3, -8]
    data.replace(special_values, np.nan, inplace=True)

    # 如果是训练集且包含目标变量，移除目标变量缺失的行
    if is_train and 'happiness' in data.columns:
        data = data.dropna(subset=['happiness'])

    # 填充缺失值和数据类型转换
    for column in data.columns:
        if column in categorical_columns and column in data.columns:
            # 填充分类特征的缺失值为众数，取第一个值
            mode_value = data[column].mode()[0]
            data[column] = data[column].fillna(mode_value)
            data[column] = data[column].astype('category').cat.codes
        elif column in numerical_columns and column in data.columns:
            # 确保数值特征的数据类型为数值并填充缺失值为中位数
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data[column] = data[column].fillna(data[column].median())

    end_time = time.time()  # 结束计时
    print(f"数据预处理耗时: {end_time - start_time:.2f} 秒")
    return data


# 数据平衡函数
def balance_data(train_data):
    """
    使用下采样或 SMOTETomek 平衡训练数据中的类别分布。

    参数:
    train_data (pd.DataFrame): 训练数据集，包含特征和目标变量。

    返回:
    tuple: 平衡后的特征 (X_resampled) 和目标变量 (y_resampled)。
    """
    start_time = time.time()  # 开始计时
    print("开始数据平衡...")

    X = train_data.drop(['happiness', 'id'], axis=1, errors='ignore')
    y = train_data['happiness']
    print("原始数据类别分布:\n", y.value_counts())

    # 检查少数类样本数量
    class_counts = y.value_counts()
    minority_classes = class_counts[class_counts < 6].index.tolist()
    if minority_classes:
        print(f"以下类别的样本数量过少，将使用下采样策略: {minority_classes}")
        under_sampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = under_sampler.fit_resample(X, y)
    else:
        smote_tomek = SMOTETomek(sampling_strategy='auto', smote=SMOTE(k_neighbors=min(3, y.value_counts().min() - 1)))
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    end_time = time.time()  # 结束计时
    print(f"数据平衡耗时: {end_time - start_time:.2f} 秒")
    print("数据平衡后类别分布:\n", pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled


# 模型训练函数
def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    使用 XGBoost 训练多分类模型。

    参数:
    X_train (pd.DataFrame): 训练集特征。
    y_train (pd.Series): 训练集目标变量。
    X_val (pd.DataFrame): 验证集特征。
    y_val (pd.Series): 验证集目标变量。

    返回:
    tuple: 训练好的模型、验证集预测结果和编码器映射。
    """
    start_time = time.time()  # 开始计时
    print("开始训练 XGBoost 模型...")

    # 对目标变量进行标签编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # 定义和训练 XGBoost 模型
    xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=42,
                              use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train_encoded)

    # 验证集预测
    y_val_pred_encoded = xgb_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

    print("验证集分类报告:\n", classification_report(y_val, y_val_pred))

    # 保存编码器的映射关系
    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    end_time = time.time()  # 结束计时
    print(f"模型训练耗时: {end_time - start_time:.2f} 秒")
    return xgb_model, y_val_pred, label_encoder_mapping


# 主函数
def main(train_file, test_file):
    """
    主函数用于数据预处理、平衡、模型训练和预测结果输出，记录每一步的执行时间。

    参数:
    train_file (str): 训练数据文件路径。
    test_file (str): 测试数据文件路径。
    """
    print_memory_usage("主函数开始")
    overall_start_time = time.time()  # 记录整个流程的开始时间

    # 指定分类和数值特征列
    categorical_columns = [
        'survey_type', 'province', 'gender', 'nationality', 'religion', 'religion_freq',
        'edu', 'political', 'health', 'health_problem', 'depression', 'hukou',
        'socialize', 'relax', 'learn', 'equity', 'class', 'work_exper',
        'work_status', 'work_type', 'work_manage', 'family_status', 'car', 'marital'
    ]
    numerical_columns = ['id', 'income', 'family_income']

    # 多线程执行数据预处理
    with ThreadPoolExecutor(max_workers=2) as executor:
        train_future = executor.submit(data_preprocessing, train_file, True, categorical_columns, numerical_columns)
        test_future = executor.submit(data_preprocessing, test_file, False, categorical_columns, numerical_columns)

        train_data = train_future.result()
        test_data = test_future.result()

    # 平衡训练数据集
    X_resampled, y_resampled = balance_data(train_data)

    # 划分训练集和验证集
    print("划分训练集和验证集...")
    split_start_time = time.time()  # 开始计时
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    split_end_time = time.time()  # 结束计时
    print(f"数据集划分耗时: {split_end_time - split_start_time:.2f} 秒")

    # 训练模型
    model, y_val_pred, label_encoder_mapping = train_xgboost_model(X_train, y_train, X_val, y_val)

    # 预测测试集
    test_ids = test_data['id']
    X_test = test_data.drop(columns=['id'], errors='ignore')
    print("开始测试集预测...")
    predict_start_time = time.time()  # 开始计时
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in
                   y_test_pred_encoded]
    predict_end_time = time.time()  # 结束计时
    print(f"测试集预测耗时: {predict_end_time - predict_start_time:.2f} 秒")

    # 导出预测结果
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv('../../report/xg/prediction_results.csv', index=False)
    print("预测结果已导出到 '../../report/xg/prediction_results.csv'")

    overall_end_time = time.time()  # 记录整个流程的结束时间
    print(f"整个流程总耗时: {overall_end_time - overall_start_time:.2f} 秒")


# 执行主函数
if __name__ == '__main__':
    train_file = '../../data/happiness_train.csv'
    test_file = '../../data/happiness_test.csv'
    main(train_file, test_file)
