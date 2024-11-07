import os
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns

TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
OUTPUT_FILE = '../../report/light/prediction_results.csv'


def ensure_directory_exists(file_path):
    """确保目录存在，如果不存在则创建。"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# 交叉验证评分函数
def cross_validate_model(X, y, message=""):
    print(f"=== {message} 开始交叉验证 ===")
    model = LGBMClassifier(objective='multiclass', random_state=42, verbose=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"{message} 交叉验证结束 - 平均准确率: {scores.mean():.4f}\n")
    return scores.mean()

# 主函数
def main(train_file, test_file):
    # 全程开始时间
    total_start_time = time.time()
    print("=== 开始幸福度预测模型训练和评估 ===")

    # 确保输出目录存在
    ensure_directory_exists(OUTPUT_FILE)

    # 获取特征列定义
    categorical_columns = FeatureColumns.CATEGORICAL_COLUMNS.value
    numerical_columns = FeatureColumns.NUMERICAL_COLUMNS.value
    one_hot_columns = FeatureColumns.ONE_HOT_COLUMNS.value
    low_correlation_features = FeatureColumns.LOW_CORRELATION_FEATURES.value

    # 使用多线程执行数据预处理
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交训练集和测试集的预处理任务
        train_future = executor.submit(data_preprocessing, pd.read_csv(train_file), True, categorical_columns, numerical_columns, one_hot_columns)
        test_future = executor.submit(data_preprocessing, pd.read_csv(test_file), False, categorical_columns, numerical_columns, one_hot_columns)

        # 获取预处理后的数据
        train_data = train_future.result()
        test_data = test_future.result()

    # 数据平衡
    X_resampled, y_resampled = balance_data(train_data, model_type='tree')

    # 交叉验证 - 保留所有特征
    start_time = time.time()
    score_all_features = cross_validate_model(X_resampled, y_resampled, message="保留所有特征")
    print(f"保留所有特征交叉验证耗时: {time.time() - start_time:.2f} 秒\n")

    # 交叉验证 - 删除低相关性特征
    start_time = time.time()
    X_reduced = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    score_reduced_features = cross_validate_model(X_reduced, y_resampled, message="删除低相关性特征")
    print(f"删除低相关性特征交叉验证耗时: {time.time() - start_time:.2f} 秒\n")

    # 决定最终特征集
    if score_reduced_features >= score_all_features:
        print("删除低相关性特征不会降低模型准确率，选择删除低相关性特征进行训练")
        X_final = X_reduced
    else:
        print("删除低相关性特征降低了模型准确率，选择保留所有特征进行训练")
        X_final = X_resampled
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42)

    # 模型训练和验证
    start_time = time.time()
    print("\n=== 开始 LightGBM 模型训练和验证 ===")
    model = LGBMClassifier(objective='multiclass', random_state=42, verbose=-1)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    model.fit(X_train, y_train_encoded)
    y_val_pred = label_encoder.inverse_transform(model.predict(X_val))
    print("验证集分类报告:\n", classification_report(y_val, y_val_pred))
    print(f"模型训练和验证耗时: {time.time() - start_time:.2f} 秒\n")

    # 读取测试集数据
    test_ids = test_data['id']  # 提取 id 列
    test_data = test_data.drop(columns=['id'])  # 删除 id 列进行预处理

    # 对齐测试集的列与训练集一致（若缺少列则填充0）
    test_data = test_data.reindex(columns=final_features, fill_value=0)

    # 测试集预测并保存结果
    start_time = time.time()
    print("\n=== 开始测试集预测并保存结果 ===")
    X_test = test_data[final_features].copy()
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

    # 创建结果 DataFrame
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"测试集预测并保存结果耗时: {time.time() - start_time:.2f} 秒\n")
    print(f"预测结果已导出至 {OUTPUT_FILE}")

    # 全程结束时间
    total_end_time = time.time()
    print(f"=== 幸福度预测任务完成，全程耗时: {total_end_time - total_start_time:.2f} 秒 ===")


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
