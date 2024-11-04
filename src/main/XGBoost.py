import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.combine import SMOTETomek
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
import psutil
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

import time

from src.main.lightBGM import process_chunk


def print_memory_usage(stage=""):
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 ** 2
    print(f"{stage} - 当前内存使用: {mem:.2f} MB")


def data_preprocessing(input_file, is_train=True, chunksize=100000, categorical_columns=None, train_columns=None):
    dtype_spec = {column: 'category' for column in categorical_columns}
    reader = pd.read_csv(input_file, chunksize=chunksize, dtype=dtype_spec)
    processed_chunks = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk, categorical_columns, train_columns): chunk for chunk in reader
        }
        for i, future in enumerate(future_to_chunk):
            try:
                processed_chunk = future.result()
                processed_chunks.append(processed_chunk)
                print(f"处理第 {i + 1} 个数据块完成，大小: {len(processed_chunk)} 行")
            except Exception as e:
                print(f"处理数据块时出错: {e}")

    processed_data = pd.concat(processed_chunks, ignore_index=True)
    if 'survey_time' in processed_data.columns:
        processed_data = processed_data.drop(columns=['survey_time'])
    end_time = time.time()
    print(f"数据预处理完成，数据集总大小: {len(processed_data)} 行，耗时: {end_time - start_time:.2f} 秒")
    print_memory_usage("数据预处理结束")

    return processed_data


def balance_data(train_data):
    print("开始数据平衡...")
    X = train_data.drop(['happiness', 'id'], axis=1, errors='ignore')
    y = train_data['happiness']
    print("原始数据类别分布:\n", y.value_counts())

    smote = SMOTE(sampling_strategy='auto', k_neighbors=NearestNeighbors(n_jobs=-1))
    tomek = TomekLinks()
    smote_tomek = SMOTETomek(smote=smote, tomek=tomek)

    start_time = time.time()
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    end_time = time.time()
    print("数据平衡完成，平衡后类别分布:\n", pd.Series(y_resampled).value_counts())
    print(f"数据平衡耗时: {end_time - start_time:.2f} 秒")
    print_memory_usage("数据平衡结束")
    return X_resampled, y_resampled


def train_xgboost_model(X_train, y_train, X_val, y_val):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_),
                                  eval_metric='mlogloss', random_state=42)

    start_time = time.time()
    xgb_model.fit(X_train, y_train_encoded)
    end_time = time.time()
    print(f"模型训练完成，耗时: {end_time - start_time:.2f} 秒")

    y_val_pred_encoded = xgb_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

    print("验证集分类报告:\n", classification_report(y_val, y_val_pred))
    print_memory_usage("模型训练结束")
    return xgb_model, y_val_pred


def main(train_file, test_file):
    print_memory_usage("主函数开始")

    print("预处理训练数据集...")
    categorical_columns = ['survey_type', 'province', 'city', 'county', 'gender', 'nationality', 'religion',
                           'house', 'car', 'marital', 'status_peer', 'status_3_before', 'view']
    train_data = data_preprocessing(train_file, is_train=True, categorical_columns=categorical_columns)

    train_columns = train_data.drop(['happiness', 'id'], axis=1, errors='ignore').columns

    test_data = pd.read_csv(test_file, dtype={col: 'category' for col in categorical_columns})
    test_ids = test_data['id']
    test_data = data_preprocessing(test_file, is_train=False, categorical_columns=categorical_columns,
                                   train_columns=train_columns)

    X_resampled, y_resampled = balance_data(train_data)

    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    print("训练 XGBoost 模型...")
    model, y_val_pred = train_xgboost_model(X_train, y_train, X_val, y_val)

    print("进行测试数据集预测...")
    start_time = time.time()
    X_test = test_data.drop(columns=['id'], errors='ignore')
    y_test_pred = model.predict(X_test)
    end_time = time.time()
    print(f"测试集预测完成，耗时: {end_time - start_time:.2f} 秒")

    print("预测结果样例:\n", pd.DataFrame({'id': test_ids[:5], 'happiness': y_test_pred[:5]}))

    print("导出预测结果到 CSV 文件...")
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv('../../report/xg/prediction_results.csv', index=False)
    print("预测结果已导出到 '../../report/xg/prediction_results.csv'")
    print_memory_usage("主函数结束")

# 通过检查 __name__ 是否等于 '__main__' 来决定是否运行 main()
if __name__ == '__main__':
    train_file = '../../data/happiness_train_abbr.csv'
    test_file = '../../data/happiness_test_abbr.csv'
    main(train_file, test_file)
