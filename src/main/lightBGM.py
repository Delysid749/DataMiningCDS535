import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.combine import SMOTETomek
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
import psutil

# 打印当前内存使用情况的函数
def print_memory_usage(stage=""):
    process = psutil.Process()  # 获取当前进程信息
    mem = process.memory_info().rss / 1024 ** 2  # 将内存使用转换为 MB
    print(f"{stage} - 当前内存使用: {mem:.2f} MB")

# 数据预处理函数，使用分块读取和多线程处理以提高效率
def data_preprocessing(input_file, is_train=True, chunksize=100000, categorical_columns=None, train_columns=None):
    """
    数据预处理函数，通过分块读取和并行处理数据块来提升效率
    """
    # 指定分类列的数据类型为 'category'，以优化内存占用
    dtype_spec = {column: 'category' for column in categorical_columns}
    reader = pd.read_csv(input_file, chunksize=chunksize, dtype=dtype_spec)

    processed_chunks = []

    # 使用多线程来并行处理数据块
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk, categorical_columns, train_columns): chunk for chunk in reader
        }
        for future in future_to_chunk:
            try:
                processed_chunk = future.result()  # 获取处理后的数据块
                processed_chunks.append(processed_chunk)  # 将处理好的数据块加入列表
            except Exception as e:
                print(f"处理数据块时出错: {e}")  # 打印处理错误信息

    # 将所有处理后的数据块合并为一个完整的 DataFrame
    processed_data = pd.concat(processed_chunks, ignore_index=True)

    # 删除无显著预测意义的 `survey_time` 列
    if 'survey_time' in processed_data.columns:
        processed_data = processed_data.drop(columns=['survey_time'])

    return processed_data

# 单个数据块的处理函数，包括缺失值填充和类别编码
def process_chunk(chunk, categorical_columns, train_columns):
    """
    处理单个数据块，包括缺失值填充和类别变量编码
    """
    # 1. 将分类列转换为 object 类型，并填充缺失值为 '未知'
    for column in categorical_columns:
        chunk[column] = chunk[column].astype('object').fillna('未知')

    # 2. 处理数值列中的缺失值和极端值（inf），并使用中位数填充
    numerical_columns = chunk.select_dtypes(include=[np.number]).columns.tolist()
    chunk[numerical_columns] = chunk[numerical_columns].replace([np.inf, -np.inf], np.nan)
    chunk[numerical_columns] = chunk[numerical_columns].fillna(chunk[numerical_columns].median())

    # 3. 对类别变量进行独热编码
    chunk = pd.get_dummies(chunk, columns=categorical_columns, drop_first=True)

    # 4. 确保测试集和训练集特征列对齐
    if train_columns is not None:
        chunk = chunk.reindex(columns=train_columns, fill_value=0)

    return chunk

# 数据平衡函数，使用 SMOTE-Tomek 方法平衡训练数据集
def balance_data(train_data):
    """
    使用 SMOTE-Tomek 方法对训练数据进行平衡
    """
    print("开始数据平衡...")
    X = train_data.drop(['happiness', 'id'], axis=1, errors='ignore')  # 去除目标列和ID列，获取特征
    y = train_data['happiness']  # 提取目标列

    # 使用 SMOTE-Tomek 方法对类别进行平衡
    smote_tomek = SMOTETomek(n_jobs=-1)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    print("数据平衡完成。")
    return X_resampled, y_resampled

# 训练和评估 LightGBM 模型的函数
def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """
    使用 LightGBM 模型进行训练并计算 F1 分数
    """
    # 检查训练集目标列中的类别数量
    unique_classes_train = y_train.nunique()
    print(f"训练集类别数: {unique_classes_train}")

    # 根据类别数选择模型的目标设置
    if unique_classes_train > 2:
        # 多分类任务
        lgbm_model = lgb.LGBMClassifier(objective='multiclass', num_class=unique_classes_train,
                                        metric='multi_logloss', boosting_type='gbdt', random_state=42)
    else:
        # 二分类任务
        lgbm_model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt', random_state=42)

    # 训练模型
    lgbm_model.fit(X_train, y_train)

    # 在验证集上进行预测，并计算 F1 分数
    y_val_pred = lgbm_model.predict(X_val)
    f1_val = f1_score(y_val, y_val_pred, average='weighted')
    print("分类报告：")
    print(classification_report(y_val, y_val_pred))

    return lgbm_model

# 主函数，用于执行数据处理、平衡和模型训练及评估
def main(train_file, test_file):
    print_memory_usage("主函数开始")

    # 开始训练集的数据预处理
    print("预处理训练数据集...")
    categorical_columns = ['survey_type', 'province', 'city', 'county', 'gender', 'nationality', 'religion',
                           'house', 'car', 'marital', 'status_peer', 'status_3_before', 'view']
    train_data = data_preprocessing(train_file, is_train=True, categorical_columns=categorical_columns)

    # 获取训练集的特征列名，以便在测试集处理时保持一致
    train_columns = train_data.drop(['happiness', 'id'], axis=1, errors='ignore').columns

    # 读取并预处理测试数据集，保留 'id' 列
    test_data = pd.read_csv(test_file, dtype={col: 'category' for col in categorical_columns})
    test_ids = test_data['id']  # 保存 'id' 列
    test_data = data_preprocessing(test_file, is_train=False, categorical_columns=categorical_columns, train_columns=train_columns)

    # 对训练数据进行平衡
    X_resampled, y_resampled = balance_data(train_data)

    # 将平衡后的数据集分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # 训练 LightGBM 模型
    print("训练 LightGBM 模型...")
    model = train_lightgbm_model(X_train, y_train, X_val, y_val)

    # 在测试集上进行预测
    print("进行测试数据集预测...")
    X_test = test_data.drop(columns=['id'], errors='ignore')
    y_test_pred = model.predict(X_test)

    # 将预测结果保存到 CSV 文件
    print("导出预测结果到 CSV 文件...")
    result_df = pd.DataFrame({'id': test_ids, 'happiness': y_test_pred})
    result_df.to_csv('../../report/prediction_results.csv', index=False)
    print("预测结果已导出到 '../../report/prediction_results.csv'")

    print_memory_usage("主函数结束")

# 如果此脚本是直接执行的，则运行主函数
if __name__ == '__main__':
    train_file = '../../data/happiness_train_abbr.csv'
    test_file = '../../data/happiness_test_abbr.csv'
    main(train_file, test_file)
