import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from concurrent.futures import ThreadPoolExecutor
import os
import time
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns
from sklearn.model_selection import cross_val_score

# 文件路径和低相关性特征
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
OUTPUT_FILE = '../../report/logistic_regression/prediction_results.csv'


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def cross_validate_model(X, y, message=""):
    print(f"=== {message} 开始交叉验证 ===")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42, class_weight='balanced')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='accuracy')
    print(f"{message} 交叉验证结束 - 平均准确率: {scores.mean():.4f}\n")
    return scores.mean()


def train_logistic_regression_model(X_train, y_train, X_val, y_val):
    start_time = time.time()
    print("开始训练 Logistic 回归模型...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    model = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial', random_state=42,
                               class_weight='balanced')
    model.fit(X_train, y_train_encoded)
    y_val_pred_encoded = model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
    print("验证集分类报告:\n", classification_report(y_val, y_val_pred))
    print(f"模型训练耗时: {time.time() - start_time:.2f} 秒")
    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return model, y_val_pred, label_encoder_mapping


def main(train_file, test_file):
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

    X_resampled, y_resampled = balance_data(train_data, model_type='linear')

    # 并行执行交叉验证
    with ThreadPoolExecutor(max_workers=2) as executor:
        start_time = time.time()
        future_all_features = executor.submit(cross_validate_model, X_resampled, y_resampled, "保留所有特征")
        X_reduced = X_resampled.drop(columns=low_correlation_features, errors='ignore')
        future_reduced_features = executor.submit(cross_validate_model, X_reduced, y_resampled, "删除低相关性特征")

        score_all_features = future_all_features.result()
        score_reduced_features = future_reduced_features.result()

        print(f"保留所有特征并行交叉验证耗时: {time.time() - start_time:.2f} 秒\n")
        print(f"删除低相关性特征并行交叉验证耗时: {time.time() - start_time:.2f} 秒\n")

    if score_reduced_features >= score_all_features:
        print("删除低相关性特征不会降低模型准确率，选择删除低相关性特征进行训练")
        X_final = X_reduced
    else:
        print("删除低相关性特征降低了模型准确率，选择保留所有特征进行训练")
        X_final = X_resampled
    final_features = X_final.columns

    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(
        test_data.drop(columns=['id'], errors='ignore').reindex(columns=final_features, fill_value=0))

    model, y_val_pred, label_encoder_mapping = train_logistic_regression_model(X_train, y_train, X_val, y_val)

    test_ids = test_data['id']
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in
                   y_test_pred_encoded]

    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"预测结果已导出到 {OUTPUT_FILE}")
    print(f"整个流程总耗时: {time.time() - overall_start_time:.2f} 秒")


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
