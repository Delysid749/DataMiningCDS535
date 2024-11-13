import os
import optuna
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns
from src.common.directory_exists import ensure_directory_exists
from src.common.log_output import log_output

# 文件路径和低相关性特征
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
OUTPUT_FILE = '../../report/gradient_boosting/prediction_results.csv'
LOG_DIR = '../logs/gradient_boosting/'

ensure_directory_exists(LOG_DIR)
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna 超参数优化目标函数，返回验证集上的准确率。
    """
    # 定义超参数搜索空间
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    max_depth = trial.suggest_int("max_depth", 3, 10)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )

    # 编码标签
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    model.fit(X_train, y_train_encoded)

    # 验证集预测并计算准确率
    y_val_pred_encoded = model.predict(X_val)
    accuracy = accuracy_score(y_val_encoded, y_val_pred_encoded)
    return accuracy


def train_gradient_boosting_with_optuna(X_train, y_train, X_val, y_val):
    """
    使用Optuna优化后的超参数训练梯度提升模型。
    """
    start_time = time.time()
    log_output("开始使用Optuna优化梯度提升模型的超参数...", log_file_path)

    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=10)

    # 获取最佳超参数并重新训练模型
    best_params = study.best_params
    log_output(f"最佳超参数: {best_params}", log_file_path)

    model = GradientBoostingClassifier(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        random_state=42
    )

    # 训练最终模型
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    model.fit(X_train, y_train_encoded)
    y_val_pred_encoded = model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

    # 验证集分类报告
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_file_path)
    log_output(f"模型训练耗时: {time.time() - start_time:.2f} 秒", log_file_path)
    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return model, label_encoder_mapping


def main(train_file, test_file):
    """
    执行整个数据处理、模型训练和预测流程。
    """
    overall_start_time = time.time()
    log_output("开始幸福度预测流程", log_file_path)
    ensure_directory_exists(OUTPUT_FILE)
    categorical_columns = FeatureColumns.CATEGORICAL_COLUMNS.value
    numerical_columns = FeatureColumns.NUMERICAL_COLUMNS.value
    one_hot_columns = FeatureColumns.ONE_HOT_COLUMNS.value
    low_correlation_features = FeatureColumns.LOW_CORRELATION_FEATURES.value

    log_output("加载并预处理数据...", log_file_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        train_data_raw = pd.read_csv(train_file)
        test_data_raw = pd.read_csv(test_file)
        train_future = executor.submit(data_preprocessing, train_data_raw, True, categorical_columns, numerical_columns,
                                       one_hot_columns)
        test_future = executor.submit(data_preprocessing, test_data_raw, False, categorical_columns, numerical_columns,
                                      one_hot_columns)
        train_data = train_future.result()
        test_data = test_future.result()

    log_output("数据平衡处理...", log_file_path)
    X_resampled, y_resampled = balance_data(train_data, model_type='tree')

    # 删除低相关性特征
    log_output(f"删除低相关性特征: {low_correlation_features}", log_file_path)
    X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    final_features = X_final.columns

    # 划分训练集和验证集
    log_output("划分训练集和验证集...", log_file_path)
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42)

    # 使用Optuna优化后的梯度提升模型进行训练和验证
    model, label_encoder_mapping = train_gradient_boosting_with_optuna(X_train, y_train, X_val, y_val)

    # 测试集预测
    log_output("加载并处理测试集数据...", log_file_path)
    test_ids = test_data['id']
    X_test = test_data.drop(columns=['id'], errors='ignore')
    X_test = X_test.reindex(columns=final_features, fill_value=0)
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in
                   y_test_pred_encoded]

    # 保存预测结果
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    log_output(f"预测结果已导出到 {OUTPUT_FILE}", log_file_path)

    total_duration = time.time() - overall_start_time
    log_output(f"整个流程总耗时: {total_duration:.2f} 秒", log_file_path)


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
