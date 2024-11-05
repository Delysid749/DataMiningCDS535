import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
            mode_value = data[column].mode()[0]
            data[column] = data[column].fillna(mode_value)
            data[column] = data[column].astype('category').cat.codes
        elif column in numerical_columns and column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data[column] = data[column].fillna(data[column].median())

    return data

# 数据平衡函数
def balance_data(train_data):
    X = train_data.drop(['happiness', 'id'], axis=1, errors='ignore')
    y = train_data['happiness']
    print("原始数据类别分布:\n", y.value_counts())

    class_counts = y.value_counts()
    minority_classes = class_counts[class_counts < 6].index.tolist()
    if minority_classes:
        print(f"以下类别的样本数量过少，将使用下采样策略: {minority_classes}")
        from imblearn.under_sampling import RandomUnderSampler
        under_sampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = under_sampler.fit_resample(X, y)
    else:
        smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    print("数据平衡后类别分布:\n", pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled

# 模型训练函数
def train_svm_model(X_train, y_train, X_val, y_val):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 使用非线性核的SVM
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
    svm_model.fit(X_train, y_train_encoded)

    y_val_pred_encoded = svm_model.predict(X_val)
    y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

    print("验证集分类报告:\n", classification_report(y_val, y_val_pred))

    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return svm_model, y_val_pred, label_encoder_mapping, scaler

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

    train_data = data_preprocessing(train_file, is_train=True, categorical_columns=categorical_columns, numerical_columns=numerical_columns)

    X_resampled, y_resampled = balance_data(train_data)

    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    print("训练非线性核 SVM 模型...")
    model, y_val_pred, label_encoder_mapping, scaler = train_svm_model(X_train, y_train, X_val, y_val)

    test_data = data_preprocessing(test_file, is_train=False, categorical_columns=categorical_columns, numerical_columns=numerical_columns)
    test_ids = test_data['id']
    X_test = scaler.transform(test_data.drop(columns=['id'], errors='ignore'))

    y_test_pred_encoded = model.predict(X_test)
    y_test_pred = [list(label_encoder_mapping.keys())[list(label_encoder_mapping.values()).index(pred)] for pred in y_test_pred_encoded]

    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv('../../report/svm/prediction_results.csv', index=False)
    print("预测结果已导出到 '../../report/svm/prediction_results.csv'")

if __name__ == '__main__':
    train_file = '../../data/happiness_train.csv'
    test_file = '../../data/happiness_test.csv'
    main(train_file, test_file)