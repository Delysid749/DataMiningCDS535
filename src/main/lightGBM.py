import os
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns

# 文件路径配置
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
LOG_DIR = '../logs/lightgbm'  # 日志文件夹路径
OUTPUT_FILE = '../../report/light/prediction_results.csv'


def ensure_directory_exists(directory_path):
    """确保目录存在，如果不存在则创建。"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# 确保日志目录存在
ensure_directory_exists(LOG_DIR)

# 定义日志文件路径
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')


def log_output(message, log_only_important=False):
    """将信息输出到控制台，并根据需要选择性写入日志文件"""
    # 总是打印到控制台
    print(message)

    # 根据条件写入日志文件
    if log_only_important:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{message}\n")


def cross_validate_model(X, y, message=""):
    """使用 cross_val_score 并行执行交叉验证"""
    log_output(f"=== {message} 开始交叉验证 ===")
    model = LGBMClassifier(objective='multiclass', random_state=42, verbose=-1, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 使用 cross_val_score 并行化处理
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    mean_score = scores.mean()

    log_output(f"{message} 交叉验证结束 - 平均准确率: {mean_score:.4f}\n")
    return mean_score


# 主流程函数
def main(train_file, test_file):
    # 记录执行代码的当地时间
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_output(f"=== 本次执行时间: {local_time} ===", log_only_important=True)

    total_start_time = time.time()
    log_output("=== 开始幸福度预测模型训练和评估 ===")

    # 获取特征列定义
    categorical_columns = FeatureColumns.CATEGORICAL_COLUMNS.value
    numerical_columns = FeatureColumns.NUMERICAL_COLUMNS.value
    one_hot_columns = FeatureColumns.ONE_HOT_COLUMNS.value
    low_correlation_features = FeatureColumns.LOW_CORRELATION_FEATURES.value

    # 数据预处理（使用多线程加速）
    with ThreadPoolExecutor(max_workers=2) as executor:
        train_future = executor.submit(data_preprocessing, pd.read_csv(train_file), True, categorical_columns,
                                       numerical_columns, one_hot_columns)
        test_future = executor.submit(data_preprocessing, pd.read_csv(test_file), False, categorical_columns,
                                      numerical_columns, one_hot_columns)
        train_data = train_future.result()
        test_data = test_future.result()

    # 数据平衡
    X_resampled, y_resampled = balance_data(train_data, model_type='tree')

    # 顺序执行两个交叉验证任务并记录各自的执行时间
    all_features_start_time = time.time()
    all_features_score = cross_validate_model(X_resampled, y_resampled, "保留所有特征")
    all_features_duration = time.time() - all_features_start_time
    log_output(f"保留所有特征交叉验证耗时: {all_features_duration:.2f} 秒")

    reduced_features_start_time = time.time()
    reduced_features_score = cross_validate_model(X_resampled.drop(columns=low_correlation_features, errors='ignore'),
                                                  y_resampled, "删除低相关性特征")
    reduced_features_duration = time.time() - reduced_features_start_time
    log_output(f"删除低相关性特征交叉验证耗时: {reduced_features_duration:.2f} 秒")

    # 总交叉验证时间
    cv_duration = all_features_duration + reduced_features_duration
    log_output(f"总交叉验证耗时: {cv_duration:.2f} 秒")

    # 记录交叉验证结果到日志文件
    if reduced_features_score >= all_features_score:
        log_output("删除低相关性特征不会降低模型准确率，选择删除低相关性特征进行训练", log_only_important=True)
        X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    else:
        log_output("删除低相关性特征降低了模型准确率，选择保留所有特征进行训练", log_only_important=True)
        X_final = X_resampled
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42)

    # 模型训练和验证
    start_time = time.time()
    log_output("\n=== 开始 LightGBM 模型训练和验证 ===")
    model = LGBMClassifier(objective='multiclass', random_state=42, verbose=-1)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    model.fit(X_train, y_train_encoded)
    y_val_pred = label_encoder.inverse_transform(model.predict(X_val))

    # 记录验证集分类报告到日志文件
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_only_important=True)
    log_output(f"模型训练和验证耗时: {time.time() - start_time:.2f} 秒\n")

    # 读取并处理测试集数据
    test_ids = test_data['id']
    test_data = test_data.drop(columns=['id'])
    test_data = test_data.reindex(columns=final_features, fill_value=0)

    # 测试集预测并保存结果
    start_time = time.time()
    log_output("\n=== 开始测试集预测并保存结果 ===")
    X_test = test_data[final_features].copy()
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

    # 保存预测结果
    ensure_directory_exists(os.path.dirname(OUTPUT_FILE))
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    log_output(f"预测结果已导出至 {OUTPUT_FILE}")

    # 全程耗时统计
    total_end_time = time.time()
    log_output(f"=== 幸福度预测任务完成，全程耗时: {total_end_time - total_start_time:.2f} 秒 ===")


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
