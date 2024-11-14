import os
import optuna
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns
from src.common.directory_exists import ensure_directory_exists
from src.common.log_output import log_output
from sklearn.model_selection import cross_val_score

# 文件路径和低相关性特征
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
OUTPUT_FILE = '../../report/logistic_regression/prediction_results.csv'
LOG_DIR = '../logs/logistic_regression/'

ensure_directory_exists(LOG_DIR)
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')


def objective(trial, X_train, y_train):
    # 定义超参数空间
    C = trial.suggest_float('C', 5, 10)
    max_iter = trial.suggest_int('max_iter', 2000, 3000)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])

    # 设置 penalty
    if solver == 'lbfgs':
        penalty = 'l2'
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            penalty=penalty,
            random_state=42,
            class_weight='balanced'
        )
    else:
        penalty = trial.suggest_categorical('penalty', ['l2', 'elasticnet'])
        if penalty == 'elasticnet':
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
            model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver=solver,
                penalty=penalty,
                l1_ratio=l1_ratio,
                random_state=42,
                class_weight='balanced'
            )
        else:
            model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver=solver,
                penalty=penalty,
                random_state=42,
                class_weight='balanced'
            )

    # 使用 LabelEncoder 对 y 进行编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # 使用5折交叉验证计算平均准确率
    scores = cross_val_score(model, X_train, y_train_encoded, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()  # 返回交叉验证的平均准确率



def train_logistic_regression_with_optuna(X_train, y_train, X_val, y_val):
    """
    使用Optuna优化后的超参数训练逻辑回归模型。
    """
    start_time = time.time()
    log_output("开始使用Optuna优化逻辑回归模型的超参数...", log_file_path)

    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=10)

    # 获取最佳超参数并重新训练模型
    best_params = study.best_params
    log_output(f"最佳超参数: {best_params}", log_file_path, log_only_important=True)

    model = LogisticRegression(
        C=best_params["C"],
        max_iter=best_params["max_iter"],
        solver=best_params["solver"],
        penalty=best_params["penalty"],
        multi_class='multinomial',
        random_state=42,
        class_weight='balanced'
    )

    # 设置 `l1_ratio`，仅在 `penalty='elasticnet'` 且 `solver='saga'` 时有效
    if best_params["penalty"] == 'elasticnet' and best_params["solver"] == 'saga':
        model.set_params(l1_ratio=best_params["l1_ratio"])

    # 训练最终模型
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    model.fit(X_train, y_train_encoded)
    y_val_pred_encoded = model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

    # 验证集分类报告
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_file_path, log_only_important=True)
    log_output(f"模型训练耗时: {time.time() - start_time:.2f} 秒", log_file_path)
    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return model, label_encoder_mapping



def main(train_file, test_file):
    """
    执行整个数据处理、模型训练和预测流程。
    """
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_output(f"=== 本次执行时间: {local_time} ===", log_file_path, log_only_important=True)
    overall_start_time = time.time()
    log_output("开始幸福度预测流程", log_file_path)
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
    X_resampled, y_resampled = balance_data(train_data, model_type='linear')

    # 删除低相关性特征
    log_output(f"删除低相关性特征: {low_correlation_features}", log_file_path)
    X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42)

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(
        test_data.drop(columns=['id'], errors='ignore').reindex(columns=final_features, fill_value=0))

    # 使用Optuna优化后的逻辑回归模型进行训练和验证
    model, label_encoder_mapping = train_logistic_regression_with_optuna(X_train, y_train, X_val, y_val)

    # 测试集预测
    test_ids = test_data['id']
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in
                   y_test_pred_encoded]

    # 保存预测结果
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    log_output(f"预测结果已导出到 {OUTPUT_FILE}", log_file_path)
    log_output(f"整个流程总耗时: {time.time() - overall_start_time:.2f} 秒", log_file_path,log_only_important=True)


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
