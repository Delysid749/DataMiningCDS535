import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from concurrent.futures import ThreadPoolExecutor
import os
import time
import optuna
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns
from src.common.directory_exists import ensure_directory_exists
from src.common.log_output import log_output

# 文件路径和低相关性特征
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
OUTPUT_FILE = '../../report/gradient_boosting/prediction_results.csv'
LOG_DIR = '../logs/gradient_boosting'

ensure_directory_exists(LOG_DIR)
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')


def objective(trial, X_train, y_train, X_val, y_val):
    # 定义需要调优的超参数空间
    n_estimators = trial.suggest_int("n_estimators", 50, 150)
    learning_rate = trial.suggest_float("learning_rate", 0.1, 0.3)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    # 创建梯度提升分类器
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=42
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()

    # 模型训练
    # model.fit(X_train, y_train)
    # y_val_pred = model.predict(X_val)
    # accuracy = accuracy_score(y_val, y_val_pred)
    #
    # return accuracy  # 目标是最大化验证集上的准确率


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=10):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    # 输出最佳参数
    log_output(f"最佳超参数: {study.best_params}", log_file_path)
    log_output(f"最佳验证集准确率: {study.best_value}", log_file_path)

    return study.best_params


def train_gradient_boosting_model(X_train, y_train, X_val, y_val, best_params):
    start_time = time.time()
    log_output("使用最佳参数开始训练梯度提升模型...", log_file_path)
    log_output(f"最佳参数配置: {best_params}",log_file_path,log_only_important=True)

    model = GradientBoostingClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    log_output("模型训练完成。", log_file_path)
    training_time = time.time() - start_time
    log_output(f"训练耗时: {training_time:.2f} 秒", log_file_path, log_only_important=True)

    log_output("开始验证模型...", log_file_path)
    y_val_pred = model.predict(X_val)
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_file_path, log_only_important=True)

    return model


def main(train_file, test_file):
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_output(f"=== 本次执行时间: {local_time} ===", log_file_path, log_only_important=True)
    overall_start_time = time.time()
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

    # 进行超参数调优
    best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # 使用最佳参数训练最终模型
    model = train_gradient_boosting_model(X_train, y_train, X_val, y_val, best_params)

    log_output("加载并处理测试集数据...", log_file_path)
    test_ids = test_data['id']
    X_test = test_data.drop(columns=['id'], errors='ignore')
    X_test = X_test.reindex(columns=final_features, fill_value=0)
    y_test_pred = model.predict(X_test)

    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    log_output(f"预测结果已导出到 {OUTPUT_FILE}", log_file_path)

    total_duration = time.time() - overall_start_time
    log_output(f"整个流程总耗时: {total_duration:.2f} 秒", log_file_path)


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
