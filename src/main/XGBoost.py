import os
import pandas as pd
import warnings
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor

from src.data_process.pre_process import data_preprocessing
from src.data_process.balance import balance_data
from src.config.feature_columns import FeatureColumns


warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

TRAIN_FILE = '../../data/happiness_train.csv'
TEST_FILE = '../../data/happiness_test.csv'
OUTPUT_FILE = '../../report/xg/prediction_results.csv'
LOW_CORRELATION_FEATURES = ['survey_type', 'religion', 'work_status', 'work_type', 'work_manage']

# 确保目录存在
def ensure_directory_exists(file_path):
    """确保目录存在，如果不存在则创建。"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


from sklearn.preprocessing import LabelEncoder


# 交叉验证评分函数
def cross_validate_model(X, y, message=""):
    print(f"=== {message} 开始交叉验证 ===")

    # 将标签转换为整数编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=42,
                          use_label_encoder=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='accuracy')
    print(f"{message} 交叉验证结束 - 平均准确率: {scores.mean():.4f}\n")
    return scores.mean()


# 模型训练函数
def train_xgboost_model(X_train, y_train, X_val, y_val):
    start_time = time.time()  # 开始计时
    print("开始训练 XGBoost 模型...")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=42, eval_metric='mlogloss', verbosity=0)
    xgb_model.fit(X_train, y_train_encoded)

    y_val_pred_encoded = xgb_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

    print("验证集分类报告:\n", classification_report(y_val, y_val_pred))

    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    end_time = time.time()  # 结束计时
    print(f"模型训练耗时: {end_time - start_time:.2f} 秒")
    return xgb_model, y_val_pred, label_encoder_mapping

# 主函数
def main(train_file, test_file):
    overall_start_time = time.time()  # 记录整个流程的开始时间

    ensure_directory_exists(OUTPUT_FILE)

    categorical_columns = FeatureColumns.CATEGORICAL_COLUMNS.value
    numerical_columns = FeatureColumns.NUMERICAL_COLUMNS.value
    one_hot_columns = FeatureColumns.ONE_HOT_COLUMNS.value

    with ThreadPoolExecutor(max_workers=2) as executor:
        train_data_raw = pd.read_csv(train_file)
        test_data_raw = pd.read_csv(test_file)

        train_future = executor.submit(data_preprocessing, train_data_raw, True, categorical_columns, numerical_columns, one_hot_columns)
        test_future = executor.submit(data_preprocessing, test_data_raw, False, categorical_columns, numerical_columns, one_hot_columns)

        train_data = train_future.result()
        test_data = test_future.result()

    # 数据平衡
    X_resampled, y_resampled = balance_data(train_data, model_type='tree')

    # 交叉验证 - 保留所有特征
    start_time = time.time()
    score_all_features = cross_validate_model(X_resampled, y_resampled, message="保留所有特征")
    print(f"保留所有特征交叉验证耗时: {time.time() - start_time:.2f} 秒\n")

    # 交叉验证 - 删除低相关性特征
    start_time = time.time()
    X_reduced = X_resampled.drop(columns=LOW_CORRELATION_FEATURES, errors='ignore')
    score_reduced_features = cross_validate_model(X_reduced, y_resampled, message="删除低相关性特征")
    print(f"删除低相关性特征交叉验证耗时: {time.time() - start_time:.2f} 秒\n")

    # 决定最终特征集
    if score_reduced_features >= score_all_features:
        print("删除低相关性特征不会降低模型准确率，选择删除低相关性特征进行训练")
        X_final = X_reduced
    else:
        print("删除低相关性特征降低了模型准确率，选择保留所有特征进行训练")
        X_final = X_resampled
    final_features = X_final.columns

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_final, y_resampled, test_size=0.2, random_state=42)

    # 模型训练和验证
    start_time = time.time()
    model, y_val_pred, label_encoder_mapping = train_xgboost_model(X_train, y_train, X_val, y_val)

    # 预测测试集
    test_ids = test_data['id']
    X_test = test_data.drop(columns=['id'], errors='ignore')
    X_test = X_test.reindex(columns=final_features, fill_value=0)

    # 测试集预测并保存结果
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in y_test_pred_encoded]

    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"预测结果已导出到 {OUTPUT_FILE}")

    overall_end_time = time.time()
    print(f"整个流程总耗时: {overall_end_time - overall_start_time:.2f} 秒")

if __name__ == '__main__':
    main(TRAIN_FILE, TEST_FILE)
