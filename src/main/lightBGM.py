import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import psutil

# 内存使用监控
def print_memory_usage(stage=""):
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 ** 2
    print(f"{stage} - 当前内存使用: {mem:.2f} MB")

# 数据预处理函数
def data_preprocessing(input_file, is_train=True, categorical_columns=None, numerical_columns=None):
    dtype_spec = {column: 'category' for column in categorical_columns}
    data = pd.read_csv(input_file, dtype=dtype_spec)

    # 将特殊值视为缺失值
    special_values = [-1, -2, -3, -8]
    data.replace(special_values, np.nan, inplace=True)

    # 如果是训练集且包含目标变量，移除目标变量缺失的行
    if is_train and 'happiness' in data.columns:
        data = data.dropna(subset=['happiness'])

    # 填充缺失值和数据类型转换
    for column in data.columns:
        if column in categorical_columns and column in data.columns:
            # 填充分类特征的缺失值为众数，取第一个值
            mode_value = data[column].mode()[0]
            data[column] = data[column].fillna(mode_value)
            data[column] = data[column].astype('category').cat.codes
        elif column in numerical_columns and column in data.columns:
            # 检查是否存在文本数据，确保数据类型一致
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data[column] = data[column].fillna(data[column].median())

    return data

# 数据平衡函数
def balance_data(train_data):
    X = train_data.drop(['happiness', 'id'], axis=1, errors='ignore')
    y = train_data['happiness']
    print("原始数据类别分布:\n", y.value_counts())

    # 检查少数类样本数量
    class_counts = y.value_counts()
    minority_classes = class_counts[class_counts < 6].index.tolist()
    if minority_classes:
        print(f"以下类别的样本数量过少，将使用下采样策略: {minority_classes}")
        under_sampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = under_sampler.fit_resample(X, y)
    else:
        smote_tomek = SMOTETomek(sampling_strategy='auto', smote=SMOTE(k_neighbors=min(3, y.value_counts().min() - 1)))
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    print("数据平衡后类别分布:\n", pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled

# 模型训练函数
def train_lightgbm_model(X_train, y_train, X_val, y_val):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    lgbm_model = LGBMClassifier(objective='multiclass', num_class=len(label_encoder.classes_), random_state=42)
    lgbm_model.fit(X_train, y_train_encoded)

    y_val_pred_encoded = lgbm_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

    print("验证集分类报告:\n", classification_report(y_val, y_val_pred))

    # 保存编码器映射
    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return lgbm_model, y_val_pred, label_encoder_mapping

# 主函数
def main(train_file, test_file):
    print_memory_usage("主函数开始")

    categorical_columns = [
        'survey_type', 'province', 'gender', 'nationality', 'religion', 'religion_freq',
        'edu', 'political', 'health', 'health_problem', 'depression', 'hukou',
        'socialize', 'relax', 'learn', 'equity', 'class', 'work_exper',
        'work_status', 'work_type', 'work_manage', 'family_status', 'car', 'marital'
    ]

    numerical_columns = ['id', 'income', 'family_income']

    # 预处理训练数据集
    train_data = data_preprocessing(train_file, is_train=True, categorical_columns=categorical_columns, numerical_columns=numerical_columns)

    # 平衡训练数据集
    X_resampled, y_resampled = balance_data(train_data)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # 训练模型
    print("训练 LightGBM 模型...")
    model, y_val_pred, label_encoder_mapping = train_lightgbm_model(X_train, y_train, X_val, y_val)

    # 预处理测试数据集
    test_data = data_preprocessing(test_file, is_train=False, categorical_columns=categorical_columns, numerical_columns=numerical_columns)
    test_ids = test_data['id']
    X_test = test_data.drop(columns=['id'], errors='ignore')

    # 测试集预测
    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in y_test_pred_encoded]

    # 导出预测结果
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv('../../report/light/prediction_results.csv', index=False)
    print("预测结果已导出到 '../../report/light/prediction_results.csv'")

# 执行主函数
if __name__ == '__main__':
    train_file = '../../data/happiness_train.csv'
    test_file = '../../data/happiness_test.csv'
    main(train_file, test_file)
