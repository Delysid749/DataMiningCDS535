import os

import optuna
import pandas as pd
import time
from sklearn.model_selection import cross_val_score
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns
from src.common.directory_exists import ensure_directory_exists
from src.common.log_output import log_output

# 文件路径和低相关性特征
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
LOG_DIR = '../logs/rf'
OUTPUT_FILE = '../../report/rf/prediction_results.csv'

# 确保日志目录存在
ensure_directory_exists(LOG_DIR)
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')


def objective(trial, X_train, y_train, X_val, y_val):
    """
       定义Optuna的超参数优化目标函数，返回验证集上的F1分数。

       参数:
       trial: optuna.trial.Trial
           Optuna中的trial对象，用于建议和管理超参数。
       X_train: pd.DataFrame
           训练集的特征数据。
       y_train: pd.Series
           训练集的目标标签。
       X_val: pd.DataFrame
           验证集的特征数据。
       y_val: pd.Series
           验证集的目标标签。

       返回:
       float
           验证集上的F1分数。
       """
    # 定义超参数搜索空间
    n_estimators = trial.suggest_int('n_estimators', 100, 400)
    max_depth = trial.suggest_int("max_depth", 15, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 6)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)
    max_features = trial.suggest_float("max_features", 0.5, 0.8)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    oob_score = trial.suggest_categorical("oob_score", [True, False]) if bootstrap else False

    # 创建随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        class_weight='balanced',
        criterion=criterion,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=-1
    )

    # 编码目标标签
    label_encoder = LabelEncoder()
    y_train_edcoded = label_encoder.fit_transform(y_train)

    scores = cross_val_score(rf_model, X_train, y_train_edcoded, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

    # y_val_encoded = label_encoder.fit_transform(y_val)
    #
    # rf_model.fit(X_train, y_train_edcoded)
    #
    # y_val_pred_encoded = rf_model.predict(X_val)
    # accuracy = accuracy_score(y_val_encoded,y_val_pred_encoded)
    #
    # return accuracy


def train_random_forest_model_with_optuna(X_train, y_train, X_val, y_val):
    """
    使用Optuna优化后的超参数训练随机森林模型。

    参数:
    X_train: pd.DataFrame
        训练集的特征数据。
    y_train: pd.Series
        训练集的目标标签。
    X_val: pd.DataFrame
        验证集的特征数据。
    y_val: pd.Series
        验证集的目标标签。

    返回:
    tuple
        - RandomForestClassifier: 最优参数下训练的随机森林模型。
        - y_val_pred: 验证集的预测标签。
        - label_encoder_mapping: 标签编码映射字典。
    """
    start_time = time.time()
    log_output("开始使用Optuna优化随机森林模型的超参数...", log_file_path)

    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=15)

    # 获取最佳超参数并重新训练模型
    best_params = study.best_params
    log_output(f"最佳超参数: {best_params}", log_file_path, log_only_important=True)

    rf_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        random_state=42,
        class_weight='balanced'
    )

    # 编码标签
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # 训练最终模型
    rf_model.fit(X_train, y_train_encoded)

    # 验证集预测并生成分类报告
    y_val_pred_encoded = rf_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_file_path, log_only_important=True)
    log_output(f"模型训练耗时: {time.time() - start_time:.2f} 秒", log_file_path,log_only_important=True)

    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return rf_model, y_val_pred, label_encoder_mapping


def main(train_file, test_file):
    """
    执行整个数据处理、模型训练和预测流程。

    参数:
    train_file: str
        训练集文件的路径。
    test_file: str
        测试集文件的路径。
    """
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_output(f"=== 本次执行时间: {local_time} ===", log_file_path, log_only_important=True)

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
    log_output(f"删除低相关性特征: {low_correlation_features}", log_file_path)
    X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42,
                                                      stratify=y_resampled)

    # 使用Optuna优化后的随机森林模型进行训练和验证
    model, y_val_pred, label_encoder_mapping = train_random_forest_model_with_optuna(X_train, y_train, X_val, y_val)

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
    log_output(f"预测结果已导出到 {OUTPUT_FILE}", log_file_path)
    log_output(f"整个流程总耗时: {time.time() - overall_start_time:.2f} 秒", log_file_path, log_only_important=True)


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
