import os
import optuna
import pandas as pd
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns
from src.common.directory_exists import ensure_directory_exists
from src.common.log_output import log_output

# 定义数据文件路径和日志输出目录
TRAIN_FILE = '../../data/happiness_train.csv'  # 训练集路径
TEST_FILE = '../../data/happiness_test.csv'  # 测试集路径
LOG_DIR = '../logs/xgboost'  # 日志输出路径
OUTPUT_FILE = '../../report/xg/prediction_results.csv'  # 预测结果报告路径
ensure_directory_exists(LOG_DIR)  # 确保日志目录存在，如果不存在则创建
log_file_path = os.path.join(LOG_DIR, 'execution_log.txt')  # 日志文件路径


def objective(trial, X_train, y_train, X_val, y_val):
    """
    定义Optuna超参数优化目标函数，通过验证集上的准确率来评估模型效果。

    参数:
    trial: optuna.trial.Trial
        Optuna用于参数调优的trial对象，用于建议和管理超参数。
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
        验证集上的模型准确率。
    """

    # 初始化标签编码器，将类别标签转换为数值编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)  # 对训练标签进行编码
    y_val_encoded = label_encoder.transform(y_val)  # 对验证标签进行编码

    # 定义模型参数，通过Optuna的suggest方法动态搜索最优超参数
    params = {
        "objective": "multi:softmax",  # 目标函数为多分类问题
        "num_class": len(label_encoder.classes_),  # 类别数量，根据训练数据中的类别数量确定
        "random_state": 42,  # 随机种子，用于保证实验的可重复性
        "eval_metric": ["mlogloss", "merror"],  # 评估指标：多分类的log损失和错误率
        "max_depth": trial.suggest_int("max_depth", 15, 18),  # 树的最大深度，控制模型复杂度
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, step=0.01),  # 学习率，控制每次迭代的步长
        "n_estimators": trial.suggest_int("n_estimators", 400, 500),  # 树的数量
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),  # 样本采样比例
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),  # 特征采样比例
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5, log=True),  # L1正则化
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5, log=True)  # L2正则化
    }

    # 使用当前超参数训练模型
    xgb_model = XGBClassifier(**params)

    scores = cross_val_score(xgb_model, X_train, y_train_encoded, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()
    # xgb_model.fit(X_train, y_train_encoded)

    # 验证集预测并计算准确率
    # y_val_pred_encoded = xgb_model.predict(X_val)
    # accuracy = (y_val_pred_encoded == y_val_encoded).mean()
    # return accuracy


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    训练XGBoost模型并使用Optuna进行超参数优化。

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
        - XGBClassifier: 最优参数下训练的XGBoost模型。
        - y_val_pred: 验证集的预测标签。
        - label_encoder_mapping: 标签编码映射字典。
    """

    start_time = time.time()
    log_output("开始训练 XGBoost 模型并进行超参数调优...", log_file_path)

    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=15)

    log_output(f"最佳参数: {study.best_params}", log_file_path, log_only_important=True)

    # 使用最佳超参数重新训练模型
    best_params = study.best_params
    best_params.update({
        "objective": "multi:softmax",
        "num_class": len(LabelEncoder().fit(y_train).classes_),  # 类别数
        "random_state": 42
    })
    xgb_model = XGBClassifier(**best_params)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)  # 编码训练标签
    y_val_encoded = label_encoder.transform(y_val)  # 编码验证标签

    # 训练最终模型
    xgb_model.fit(
        X_train,
        y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        verbose=False
    )

    # 验证集预测并生成分类报告
    y_val_pred_encoded = xgb_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
    classification_report_output = classification_report(y_val, y_val_pred)
    log_output("验证集分类报告:\n" + classification_report_output, log_file_path, log_only_important=True)
    log_output(f"模型训练耗时: {time.time() - start_time:.2f} 秒", log_file_path)

    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return xgb_model, y_val_pred, label_encoder_mapping


def main(train_file, test_file):
    """
    执行XGBoost模型的训练和测试过程，包含数据预处理、模型训练、验证和结果输出。

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
    categorical_columns = FeatureColumns.CATEGORICAL_COLUMNS.value  # 类别特征列
    numerical_columns = FeatureColumns.NUMERICAL_COLUMNS.value  # 数值特征列
    one_hot_columns = FeatureColumns.ONE_HOT_COLUMNS.value  # 独热编码列
    low_correlation_features = FeatureColumns.LOW_CORRELATION_FEATURES.value  # 低相关性特征

    with ThreadPoolExecutor(max_workers=2) as executor:
        # 并行预处理训练和测试数据
        train_data_raw = pd.read_csv(train_file)
        test_data_raw = pd.read_csv(test_file)
        train_future = executor.submit(data_preprocessing, train_data_raw, True, categorical_columns, numerical_columns,
                                       one_hot_columns)
        test_future = executor.submit(data_preprocessing, test_data_raw, False, categorical_columns, numerical_columns,
                                      one_hot_columns)
        train_data = train_future.result()
        test_data = test_future.result()

    # 处理平衡数据
    X_resampled, y_resampled = balance_data(train_data, model_type='tree')

    # 删除低相关性特征
    log_output(f"删除低相关性特征: {low_correlation_features}", log_file_path)
    X_final = X_resampled.drop(columns=low_correlation_features, errors='ignore')
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.3, random_state=42,
                                                      stratify=y_resampled)

    # 模型训练和验证
    model, y_val_pred, label_encoder_mapping = train_xgboost_model(X_train, y_train, X_val, y_val)

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
