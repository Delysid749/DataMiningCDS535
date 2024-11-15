- DataMiningCDS535 Project

  ## 项目简介

  本项目围绕幸福指数数据集展开，目标是通过多种机器学习模型来预测幸福感，并分析各特征对幸福感的影响。项目包括数据预处理、特征工程、特征选择、模型训练、超参数优化、结果评估和可视化分析。所有代码文件都经过精心设计，以确保可复现性和模块化。

  ## 目录结构

  ```
  bash
  
  
  复制代码
  DataMiningCDS535/
  ├── data/                         # 原始和处理后的数据文件
  ├── report/                       # 各模型的分析报告与结果
  ├── src/                          # 源代码文件
  ├── logs/                         # 日志文件
  ├── README.md                     # 项目说明文件
  └── 开发文档.docx                 # 开发文档
  ```

  ## 数据文件详细说明

  ### `data` 文件夹

  - **happiness_index.xlsx**: 包含幸福感问卷中所有变量的详细说明，包括变量名、编号、问题内容和取值含义。此文件为数据字典，帮助理解数据集中每个字段的具体含义。
  - **happiness_submit.csv**: 提交文件，包含预测的幸福感分数，用于评估模型的实际应用效果。
  - **happiness_test.csv**: 测试数据集，不包含目标变量 `happiness`，用于模型评估的泛化能力测试。
  - **happiness_train.csv**: 训练数据集，包含用户的各类特征和幸福感评分，作为模型学习的基础数据。
  - **processed_test.csv** 和 **processed_train.csv**: 预处理后的测试和训练数据集，经过缺失值处理、异常值处理、数值归一化和类别编码，使数据更适合模型训练和预测。

  ## 工作流程与代码说明

  ### 1. 数据预处理

  #### 代码文件

  - **balance.py**: 平衡数据集中的各类样本数量，使用过采样或欠采样等方法，防止类别不平衡导致的模型偏差。例如，如果某一类别的样本数远少于其他类别，则在模型训练时可能导致其无法正确分类。
  - **pre_process.py**: 包含数据清洗与预处理的核心步骤，包括：
    - **缺失值处理**：将特定的数值（如 `-1`, `-2`, `-3`, `-8` 等）替换为 `NaN` 以便于识别缺失值。
    - **数值归一化**：对数值型特征进行标准化或归一化，确保不同尺度的数据不会对模型造成误导。
    - **类别编码**：对分类特征进行编码，如独热编码（One-Hot Encoding），将文本数据转换为模型可以理解的数值格式。

  ### 2. 特征工程与选择

  #### 特征分析与选择代码

  - **feature_columns.py**: 定义了特征的分类和枚举，包含：
    - `CATEGORICAL_COLUMNS`: 分类特征列表，使用独热编码。
    - `NUMERICAL_COLUMNS`: 数值特征列表，适用数值归一化。
    - `ONE_HOT_COLUMNS`: 需要进行独热编码的特征。
    - `LOW_CORRELATION_FEATURES`: 低相关性特征列表，供特征选择时参考。
  - **categorical_correlation.py**: 使用 `chi2` 和 `mutual_info_classif` 方法分析分类特征的重要性：
    - **卡方检验**：评估特征与目标变量的相关性，卡方检验适用于离散型特征。
    - **互信息**：计算特征与目标变量的互信息，反映非线性关系。
    - **自适应加权组合得分**：通过设置阈值来确定特征的相关性，并根据其重要性分类为高、中、低三个等级。
  - **numerical_correlation.py**: 对数值特征执行相关性分析，生成相关性矩阵，帮助识别特征间的线性相关性，并利用散点图进一步展示各特征之间的关系，指导特征选择过程。

  #### 描述性统计分析代码

  - **descriptive_statistics_categorical.py**: 生成每个分类特征的值分布图，用于分析各特征的类别频率分布，帮助理解特征的取值范围和集中程度。
  - **descriptive_statistics_numerical.py**: 生成数值特征的描述性统计数据，包括均值、中位数、标准差等，同时生成数据分布图（如直方图、核密度图）以便识别数据分布特征和异常值。

  ### 3. 模型训练、验证与测试

  #### 模型实现与详细说明

  本项目采用了五种机器学习模型对幸福感数据进行预测，每种模型的具体设置、超参数调整以及在数据集上的效果评估如下：

  1. **Gradient Boosting** (`GradientBoostingClassifier.py`)

     - **模型特点**：Gradient Boosting 是一种集成方法，通过多棵弱学习器（通常为决策树）的加权组合提升预测效果。每棵树的训练都会基于前一棵树的残差，优化模型的拟合能力。
     - **超参数设置**：调整的超参数包括 `n_estimators`（树的数量）、`learning_rate`（学习率）、`max_depth`（树的最大深度）、`min_samples_split` 和 `min_samples_leaf`（控制每棵树的分裂条件）、`subsample`（子样本比例）。
     - **效果评估**：使用验证集计算每类的 `precision`、`recall` 和 `f1-score`，并根据准确率判断模型效果。具体结果和分类报告可以在 `logs/gradient_boosting/execution_log.txt` 中查看(execution_log)。

  2. **LightGBM** (`lightGBM.py`)

     - **模型特点**：LightGBM 是一种高效的梯度提升框架，基于直方算法构建树结构，能够处理大规模数据。其特有的 `leaf-wise` 增长策略能更快收敛，但需要合适的参数调整以防止过拟合。
     - **超参数设置**：包括 `num_leaves`（叶节点数）、`max_depth`、`learning_rate`、`feature_fraction`（特征选择比例）和 `bagging_fraction`（数据抽样比例）。通过交叉验证确定最佳参数组合。
     - **效果评估**：通过验证集计算分类报告和整体准确率，报告内容记录于 `logs/lightgbm/execution_log.txt` 中(execution_log)。

  3. **随机森林** (`Random_forest.py`)

     - **模型特点**：随机森林是一种集成方法，利用多棵决策树的投票结果进行分类。它通过随机选择特征和样本提高了模型的鲁棒性，能有效防止过拟合。
     - **超参数设置**：包括 `n_estimators`（树的数量）、`max_depth`、`min_samples_split`、`min_samples_leaf` 和 `max_features`（分裂时考虑的最大特征数）。参数优化通过网格搜索完成。
     - **效果评估**：验证集上每类的 `precision`、`recall` 和 `f1-score`，并记录在 `logs/rf/execution_log.txt`(execution_log)。

  4. **逻辑回归** (`logistic_regression.py`)

     - **模型特点**：逻辑回归是一种线性模型，适用于二分类问题。通过 sigmoid 函数将线性回归结果转换为概率输出，在多分类问题中采用 `softmax` 方案。
     - **超参数设置**：主要调整 `C`（正则化强度），`solver`（优化算法），`penalty`（正则化类型）和 `max_iter`（最大迭代次数）。不同正则化类型（如 L1、L2）适用于不同的稀疏度需求。
     - **效果评估**：在验证集上计算分类报告，评估每类的 `precision`、`recall` 和 `f1-score`。日志记录于 `logs/logistic_regression/execution_log.txt`(execution_log)。

  5. **XGBoost** (`XGBoost.py`)

     - **模型特点**：XGBoost 是一种增强型的梯度提升树模型，通过正则化进一步提升模型的泛化能力。其特有的 `column sampling` 增加了模型的鲁棒性。
     - **超参数设置**：包括 `max_depth`、`learning_rate`、`n_estimators`、`subsample`、`colsample_bytree` 以及正则化参数 `reg_alpha` 和 `reg_lambda`。通过网格搜索或随机搜索实现最优组合。
     - **效果评估**：XGBoost 模型的表现通过验证集分类报告来衡量。日志文件 `logs/xgboost/execution_log.txt` 中详细记录了验证集上的 `precision`、`recall` 和 `f1-score`，以及每次实验的最佳超参数配置(execution_log)。

     #### 日志文件内容与记录

     每个模型的训练与验证过程均生成对应的日志文件，以 `execution_log.txt` 命名。这些日志文件包括以下内容：

     - **训练耗时**：每次模型训练的总耗时，用于评估模型的训练效率。例如，Gradient Boosting 的某次训练耗时记录为 "训练耗时: 69.06 秒"。

     - **最佳参数配置**：经过超参数优化后，记录每次实验的最佳参数。例如，XGBoost 模型的最佳参数配置包含 `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_alpha` 和 `reg_lambda` 等。

     - **分类报告**：每次验证集的 `precision`、`recall` 和 `f1-score`，用于评估模型在各类别上的表现。详细分类报告帮助理解模型在不同类别上的准确性和召回率，如下所示：

       ```
       plaintext
       
       
       复制代码
       验证集分类报告:
                     precision    recall  f1-score   support
       
                1.0       0.97      0.99      0.98      2417
                2.0       0.86      0.89      0.88      2476
                3.0       0.77      0.74      0.76      2484
                4.0       0.67      0.76      0.71      2416
                5.0       0.83      0.72      0.77      2517
       
           accuracy                           0.82     12310
          macro avg       0.82      0.82      0.82     12310
       weighted avg       0.82      0.82      0.82     12310
       ```

     - **整体准确率**：每次模型训练的总体准确率，用于对比各模型的整体表现。例如，LightGBM 模型的整体准确率为 0.87，有助于对比该模型在不同超参数配置下的表现差异。

     ### 4. 可视化分析

     在 `report` 文件夹中，每个模型生成两个关键输出文件，以便于更直观地理解模型的性能和特征重要性：

     - **happiness_correlation.png**：该文件展示了模型预测结果的幸福感相关性，显示特征与目标变量的相关性强度。通过柱状图的形式直观展示出最重要的前 10 个特征。
     - **prediction_results.csv**：保存模型的预测结果，包含每个样本的 `id` 和预测的 `happiness` 分数，用于后续的结果分析和提交。

     ### 5. 最终分析与结果汇总

     - **final_analysis.py**：综合多个模型的预测结果和相关性分析，评估不同特征在幸福感预测中的重要性，并比较各模型的性能。在 `final_analysis.py` 中，将各模型的结果汇总，以便决策者对幸福感的主要影响因素有更深入的了解。