import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns

# 文件路径和低相关性特征
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
OUTPUT_FILE = '../../report/rf/prediction_results.csv'


# 确保目录存在
def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# 交叉验证评分函数
def cross_validate_model(X, y, message=""):
    print(f"=== {message} 开始交叉验证 ===")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='accuracy')
    print(f"{message} 交叉验证结束 - 平均准确率: {scores.mean():.4f}\n")
    return scores.mean()


# 模型训练函数
def train_random_forest_model(X_train, y_train, X_val, y_val):
    start_time = time.time()
    print("开始训练随机森林模型...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train_encoded)
    y_val_pred_encoded = rf_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
    print("验证集分类报告:\n", classification_report(y_val, y_val_pred))
    print(f"模型训练耗时: {time.time() - start_time:.2f} 秒")
    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return rf_model, y_val_pred, label_encoder_mapping


# 主函数
def main(train_file, test_file):
    overall_start_time = time.time()
    ensure_directory_exists(OUTPUT_FILE)
    categorical_columns = FeatureColumns.CATEGORICAL_COLUMNS.value
    numerical_columns = FeatureColumns.NUMERICAL_COLUMNS.value
    one_hot_columns = FeatureColumns.ONE_HOT_COLUMNS.value
    low_correlation_features = FeatureColumns.LOW_CORRELATION_FEATURES.value

    # 数据预处理
    with ThreadPoolExecutor(max_workers=2) as executor:
        train_data_raw = pd.read_csv(train_file)
        test_data_raw = pd.read_csv(test_file)
        train_future = executor.submit(data_preprocessing, train_data_raw, True, categorical_columns, numerical_columns,
                                       one_hot_columns)
        test_future = executor.submit(data_preprocessing, test_data_raw, False, categorical_columns, numerical_columns,
                                      one_hot_columns)
        train_data = train_future.result()
        test_data = test_future.result()

    # 数据平衡
    X_resampled, y_resampled = balance_data(train_data, model_type='tree')

    # 交叉验证并行执行
    with ThreadPoolExecutor(max_workers=2) as executor:
        start_time_cv = time.time()  # 记录交叉验证的总开始时间
        cv_start_times = {"all_features": time.time(), "reduced_features": time.time()}  # 初始化每个任务的开始时间

        futures = {
            executor.submit(cross_validate_model, X_resampled, y_resampled, "保留所有特征"): "all_features",
            executor.submit(cross_validate_model, X_resampled.drop(columns=low_correlation_features, errors='ignore'),
                            y_resampled, "删除低相关性特征"): "reduced_features"
        }

        results = {}
        for future in as_completed(futures):
            label = futures[future]
            cv_duration = time.time() - cv_start_times[label]  # 计算每个任务的真实耗时
            results[label] = future.result()
            print(f"{label} 交叉验证耗时: {cv_duration:.2f} 秒")

        # 记录并打印总的交叉验证耗时
        total_cv_duration = time.time() - start_time_cv
        print(f"交叉验证总耗时: {total_cv_duration:.2f} 秒")

    # 比较交叉验证结果以确定最终特征集
    if results["reduced_features"] >= results["all_features"]:
        print("删除低相关性特征不会降低模型准确率，选择删除低相关性特征进行训练")
        X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    else:
        print("删除低相关性特征降低了模型准确率，选择保留所有特征进行训练")
        X_final = X_resampled
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42)

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
    print(f"预测结果已导出到 {OUTPUT_FILE}")
    print(f"整个流程总耗时: {time.time() - overall_start_time:.2f} 秒")


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
