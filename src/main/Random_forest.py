import os
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns

# 文件路径和低相关性特征
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
LOG_DIR = '../logs/rf'
OUTPUT_FILE = '../../report/rf/prediction_results.csv'


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# 确保日志目录存在
ensure_directory_exists(LOG_DIR)
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')


def log_output(message, log_only_important=False):
    """将信息输出到控制台，并根据需要选择性写入日志文件"""
    print(message)
    # 确保日志文件所在目录存在
    ensure_directory_exists(os.path.dirname(log_file_path))
    if log_only_important:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{message}\n")


def train_random_forest_model(X_train, y_train, X_val, y_val):
    start_time = time.time()
    log_output("开始训练随机森林模型...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train_encoded)
    y_val_pred_encoded = rf_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_only_important=True)
    log_output(f"模型训练耗时: {time.time() - start_time:.2f} 秒")
    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return rf_model, y_val_pred, label_encoder_mapping


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
        train_future = executor.submit(data_preprocessing, train_data_raw, True, categorical_columns, numerical_columns,
                                       one_hot_columns)
        test_future = executor.submit(data_preprocessing, test_data_raw, False, categorical_columns, numerical_columns,
                                      one_hot_columns)
        train_data = train_future.result()
        test_data = test_future.result()

    # 数据平衡处理
    X_resampled, y_resampled = balance_data(train_data, model_type='tree')

    # 删除低相关性特征
    log_output(f"删除低相关性特征: {low_correlation_features}")
    X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # 模型训练和验证
    model, y_val_pred, label_encoder_mapping = train_random_forest_model(X_train, y_train, X_val, y_val)

    # 测试集预测
    test_ids = test_data['id']
    X_test = test_data.drop(columns=['id'], errors='ignore')
    X_test = X_test.reindex(columns=final_features, fill_value=0)
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in
                   y_test_pred_encoded]

    # 保存预测结果
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    log_output(f"预测结果已导出到 {OUTPUT_FILE}")
    log_output(f"整个流程总耗时: {time.time() - overall_start_time:.2f} 秒")


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
