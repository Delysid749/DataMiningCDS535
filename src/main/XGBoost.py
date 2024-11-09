import os
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns

# 文件路径和低相关性特征
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
LOG_DIR = '../logs/xgboost'
OUTPUT_FILE = '../../report/xg/prediction_results.csv'

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

ensure_directory_exists(LOG_DIR)
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')

def log_output(message, log_only_important=False):
    """将信息输出到控制台，并根据需要选择性写入日志文件"""
    print(message)
    ensure_directory_exists(os.path.dirname(log_file_path))
    if log_only_important:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{message}\n")

def cross_validate_model(X, y, message=""):
    """执行交叉验证并对每一折并行计算，输出平均准确率"""
    log_output(f"=== {message} 开始交叉验证 ===")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model_params = {
        'objective': 'multi:softmax',
        'num_class': len(label_encoder.classes_),
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'n_jobs': 1  # 每折单独并行，因此每个模型内部使用单进程
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(train_and_score, X.iloc[train_index], X.iloc[val_index], y_encoded[train_index], y_encoded[val_index], model_params)
            for train_index, val_index in skf.split(X, y_encoded)
        ]

        for future in as_completed(futures):
            scores.append(future.result())

    mean_score = sum(scores) / len(scores)
    log_output(f"{message} 交叉验证结束 - 平均准确率: {mean_score:.4f}\n")
    return mean_score

def train_and_score(X_train, X_val, y_train, y_val, model_params):
    """训练 XGBoost 模型并返回单折的验证准确率"""
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

def train_xgboost_model(X_train, y_train, X_val, y_val):
    start_time = time.time()
    log_output("开始训练 XGBoost 模型...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train_encoded)

    y_val_pred_encoded = xgb_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_only_important=True)
    log_output(f"模型训练耗时: {time.time() - start_time:.2f} 秒")

    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return xgb_model, y_val_pred, label_encoder_mapping

def main(train_file, test_file):
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_output(f"=== 本次执行时间: {local_time} ===", log_only_important=True)
    overall_start_time = time.time()
    ensure_directory_exists(OUTPUT_FILE)
    categorical_columns = FeatureColumns.CATEGORICAL_COLUMNS.value
    numerical_columns = FeatureColumns.NUMERICAL_COLUMNS.value
    one_hot_columns = FeatureColumns.ONE_HOT_COLUMNS.value
    low_correlation_features = FeatureColumns.LOW_CORRELATION_FEATURES.value

    with ThreadPoolExecutor(max_workers=2) as executor:
        train_data_raw = pd.read_csv(train_file)
        test_data_raw = pd.read_csv(test_file)
        train_future = executor.submit(data_preprocessing, train_data_raw, True, categorical_columns, numerical_columns, one_hot_columns)
        test_future = executor.submit(data_preprocessing, test_data_raw, False, categorical_columns, numerical_columns, one_hot_columns)
        train_data = train_future.result()
        test_data = test_future.result()

    X_resampled, y_resampled = balance_data(train_data, model_type='tree')

    all_features_start_time = time.time()
    all_features_score = cross_validate_model(X_resampled, y_resampled, "保留所有特征")
    all_features_duration = time.time() - all_features_start_time
    log_output(f"保留所有特征交叉验证耗时: {all_features_duration:.2f} 秒")

    reduced_features_start_time = time.time()
    reduced_features_score = cross_validate_model(X_resampled.drop(columns=low_correlation_features, errors='ignore'), y_resampled, "删除低相关性特征")
    reduced_features_duration = time.time() - reduced_features_start_time
    log_output(f"删除低相关性特征交叉验证耗时: {reduced_features_duration:.2f} 秒")

    cv_duration = all_features_duration + reduced_features_duration
    log_output(f"总交叉验证耗时: {cv_duration:.2f} 秒")

    if reduced_features_score >= all_features_score:
        log_output("删除低相关性特征不会降低模型准确率，选择删除低相关性特征进行训练", log_only_important=True)
        X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    else:
        log_output("删除低相关性特征降低了模型准确率，选择保留所有特征进行训练", log_only_important=True)
        X_final = X_resampled
    final_features = X_final.columns

    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    model, y_val_pred, label_encoder_mapping = train_xgboost_model(X_train, y_train, X_val, y_val)

    test_ids = test_data['id']
    X_test = test_data.drop(columns=['id'], errors='ignore')
    X_test = X_test.reindex(columns=final_features, fill_value=0)
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in y_test_pred_encoded]

    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    log_output(f"预测结果已导出到 {OUTPUT_FILE}")
    log_output(f"整个流程总耗时: {time.time() - overall_start_time:.2f} 秒")

if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
