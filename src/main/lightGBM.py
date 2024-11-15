import os
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import optuna
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns
from src.common.directory_exists import ensure_directory_exists
from src.common.log_output import log_output
from sklearn.model_selection import cross_val_score
# 文件路径配置
TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
LOG_DIR = '../logs/lightgbm'  # 日志文件夹路径
OUTPUT_FILE = '../../report/light/prediction_results.csv'

# 确保日志目录存在
ensure_directory_exists(LOG_DIR)

# 定义日志文件路径
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')


# 超参数优化目标函数
def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'multiclass',
        'num_class': len(set(y_train)),
        'random_state': 42,
        'verbose': -1,
        'max_depth': trial.suggest_int("max_depth", 10, 20),
        'learning_rate': trial.suggest_float("learning_rate", 0.03, 0.15),
        'n_estimators': trial.suggest_int("n_estimators", 400, 800),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 10, log=True)
    }

    model = LGBMClassifier(**param)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    return scores.mean()  # 返回交叉验证的平均准确率


    # model.fit(X_train, y_train)
    # y_val_pred = model.predict(X_val)
    # score = f1_score(y_val, y_val_pred, average='weighted')
    # return score


# 超参数自动调优函数
def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=10)
    return study.best_params


# 主流程函数
def main(train_file, test_file):
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_output(f"=== 本次执行时间: {local_time} ===", log_file_path, log_only_important=True)

    total_start_time = time.time()
    log_output("=== 开始幸福度预测模型训练和评估 ===", log_file_path)

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

    # 删除低相关性特征
    X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42)

    # 标签编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    start_time = time.time()
    # 进行超参数优化
    log_output("\n=== 开始超参数自动调优 ===", log_file_path)
    best_params = optimize_hyperparameters(X_train, y_train_encoded, X_val, y_val_encoded)
    log_output(f"最佳超参数配置: {best_params}", log_file_path, log_only_important=True)

    # 使用最佳参数配置训练模型

    log_output("\n=== 开始 LightGBM 模型训练和验证 ===", log_file_path)
    model = LGBMClassifier(**best_params, objective='multiclass', random_state=42, verbose=-1)
    model.fit(X_train, y_train_encoded)
    y_val_pred = label_encoder.inverse_transform(model.predict(X_val))

    # 记录验证集分类报告到日志文件
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_file_path, log_only_important=True)
    log_output(f"模型训练和验证耗时: {time.time() - start_time:.2f} 秒\n", log_file_path, log_only_important=True)

    # 读取并处理测试集数据
    test_ids = test_data['id']
    test_data = test_data.drop(columns=['id'])
    test_data = test_data.reindex(columns=final_features, fill_value=0)

    # 测试集预测并保存结果
    start_time = time.time()
    log_output("\n=== 开始测试集预测并保存结果 ===", log_file_path)
    X_test = test_data[final_features].copy()
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

    # 保存预测结果
    ensure_directory_exists(os.path.dirname(OUTPUT_FILE))
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    log_output(f"预测结果已导出至 {OUTPUT_FILE}", log_file_path)

    # 全程耗时统计
    total_end_time = time.time()
    log_output(f"=== 幸福度预测任务完成，全程耗时: {total_end_time - total_start_time:.2f} 秒 ===", log_file_path,
               log_only_important=True)


if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
