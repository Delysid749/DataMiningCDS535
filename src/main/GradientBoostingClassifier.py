import pandas as pd
import numpy as np
import os
import time
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import psutil
from threading import Thread


# 内存使用监控
def print_memory_usage(stage=""):
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 ** 2
    print(f"{stage} - 当前内存使用: {mem:.2f} MB")


# 数据预处理函数
def data_preprocessing(input_file, is_train=True, categorical_columns=None, numerical_columns=None, output_list=None):
    start_time = time.time()
    print(f"{'训练' if is_train else '测试'}数据预处理开始...")

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
            mode_value = data[column].mode()[0]
            data[column] = data[column].fillna(mode_value)
            data[column] = data[column].astype('category').cat.codes
        elif column in numerical_columns and column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data[column] = data[column].fillna(data[column].median())

    print(f"{'训练' if is_train else '测试'}数据预处理完成 - 耗时: {time.time() - start_time:.2f} 秒")
    if output_list is not None:
        output_list.append(data)


# 数据平衡函数
def balance_data(X, y):
    print("使用 SMOTE + Tomek Links 平衡数据...")
    smote_tomek = SMOTETomek(random_state=42, sampling_strategy='auto')
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    print("数据平衡后类别分布:\n", pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled


# 模型训练函数 - 使用梯度提升树和 k 折交叉验证
def train_with_cross_validation(X_train, y_train):
    start_time = time.time()
    print("开始使用梯度提升树模型进行 k 折交叉验证...")

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)

    # 使用 StratifiedKFold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')

    print(f"k 折交叉验证准确率: {cross_val_scores}")
    print(f"平均交叉验证准确率: {cross_val_scores.mean():.4f}")
    print(f"交叉验证完成 - 耗时: {time.time() - start_time:.2f} 秒")

    # 在整个训练集上训练最终模型
    model.fit(X_train, y_train)
    return model


# 检查和创建文件夹
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# 主函数
def main(train_file, test_file):
    start_time_main = time.time()
    print_memory_usage("主函数开始")

    categorical_columns = [
        'survey_type', 'province', 'gender', 'nationality', 'religion', 'religion_freq',
        'edu', 'political', 'health', 'health_problem', 'depression', 'hukou',
        'socialize', 'relax', 'learn', 'equity', 'class', 'work_exper',
        'work_status', 'work_type', 'work_manage', 'family_status', 'car', 'marital'
    ]

    numerical_columns = ['id', 'income', 'family_income']

    # 使用多线程预处理数据
    train_data = []
    test_data = []
    threads = [
        Thread(target=data_preprocessing, args=(train_file, True, categorical_columns, numerical_columns, train_data)),
        Thread(target=data_preprocessing, args=(test_file, False, categorical_columns, numerical_columns, test_data))
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    train_data = train_data[0]
    test_data = test_data[0]

    # 划分训练集和验证集
    X = train_data.drop(['happiness', 'id'], axis=1, errors='ignore')
    y = train_data['happiness']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(test_data.drop(columns=['id'], errors='ignore'))

    # 平衡训练数据
    X_train_resampled, y_train_resampled = balance_data(X_train, y_train)

    # 训练模型
    print("训练梯度提升树模型...")
    model = train_with_cross_validation(X_train_resampled, y_train_resampled)

    # 验证集预测
    y_val_pred = model.predict(X_val)
    print("验证集分类报告:\n", classification_report(y_val, y_val_pred, zero_division=1))

    # 测试集预测
    test_ids = test_data['id']
    y_test_pred = model.predict(X_test)

    # 检查并创建导出文件夹
    output_dir = '../../report/gradient_boosting_with_cv/'
    ensure_directory_exists(output_dir)

    # 导出预测结果
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(f'{output_dir}prediction_results.csv', index=False)
    print(f"预测结果已导出到 '{output_dir}prediction_results.csv'")
    print(f"主函数完成 - 总耗时: {time.time() - start_time_main:.2f} 秒")


# 执行主函数
if __name__ == '__main__':
    train_file = '../../data/happiness_train.csv'
    test_file = '../../data/happiness_test.csv'
    main(train_file, test_file)
