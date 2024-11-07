# src/main/lightGBM.py

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from src.data_process.pre_process import data_preprocessing
from src.config.feature_columns import FeatureColumns


TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
OUTPUT_FILE = '../../report/light/prediction_results.csv'

# 数据平衡函数
def balance_data(train_data):
    start_time = time.time()
    print("=== 开始数据平衡 ===")
    X = train_data.drop(columns=['happiness', 'id'], errors='ignore')
    y = train_data['happiness']

    print("数据平衡前类别分布:\n", y.value_counts())
    smote_tomek = SMOTETomek(sampling_strategy='auto', smote=SMOTE(k_neighbors=min(3, y.value_counts().min() - 1)))
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    print("数据平衡后类别分布:\n", pd.Series(y_resampled).value_counts())
    print(f"数据平衡耗时: {time.time() - start_time:.2f} 秒\n")
    return X_resampled, y_resampled

# 交叉验证评分函数
def cross_validate_model(X, y, message=""):
    print(f"=== {message} 开始交叉验证 ===")
    model = LGBMClassifier(objective='multiclass', random_state=42, verbose=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"{message} 交叉验证结束 - 平均准确率: {scores.mean():.4f}\n")
    return scores.mean()


def main(train_file, test_file):
    total_start_time = time.time()

    # 获取特征列定义
    categorical_columns = FeatureColumns.CATEGORICAL_COLUMNS.value
    numerical_columns = FeatureColumns.NUMERICAL_COLUMNS.value
    one_hot_columns = FeatureColumns.ONE_HOT_COLUMNS.value
    low_correlation_features = FeatureColumns.LOW_CORRELATION_FEATURES.value

    # 训练集预处理
    train_data = pd.read_csv(train_file)
    train_data = data_preprocessing(train_data, is_train=True,
                                    categorical_columns=categorical_columns,
                                    numerical_columns=numerical_columns,
                                    one_hot_columns=one_hot_columns)

    # 数据平衡
    X_resampled, y_resampled = balance_data(train_data)

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
    test_data = pd.read_csv(test_file)
    test_ids = test_data['id']  # 提取 id 列
    test_data = test_data.drop(columns=['id'])  # 删除 id 列进行预处理

    # 测试集预处理
    test_data = data_preprocessing(test_data, is_train=False,
                                   categorical_columns=FeatureColumns.CATEGORICAL_COLUMNS.value,
                                   numerical_columns=FeatureColumns.NUMERICAL_COLUMNS.value,
                                   one_hot_columns=FeatureColumns.ONE_HOT_COLUMNS.value)

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


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
