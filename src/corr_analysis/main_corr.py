import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_happiness_correlation(test_file, prediction_results_file, output_image_file):
    """
    计算测试集特征与幸福感预测值的相关性并生成图表。

    Parameters:
    - test_file (str): 测试集数据文件路径。
    - prediction_results_file (str): 模型预测结果文件路径。
    - output_image_file (str): 输出图表文件路径。
    """
    # 加载测试集和预测结果
    happiness_test_df = pd.read_csv(test_file)
    prediction_results_df = pd.read_csv(prediction_results_file)

    # 合并预测结果到测试集数据上
    merged_df = happiness_test_df.merge(prediction_results_df, on="id", how="inner")

    # 分析特征对 happiness 的影响
    correlations = merged_df.corr()['happiness'].sort_values(ascending=False)

    # 显示幸福度与主要特征的相关性
    print("Features Correlated with Happiness:")
    print(correlations)

    # 可视化高相关性特征
    top_features = correlations.index[1:11]  # 选择与 happiness 最相关的前10个特征
    plt.figure(figsize=(12, 8))
    sns.barplot(x=correlations[top_features], y=top_features)
    plt.title("Top 10 Features Correlated with Happiness")
    plt.xlabel("Correlation with Happiness")
    plt.ylabel("Feature")

    # 保存图表
    plt.savefig(output_image_file)
    plt.show()

