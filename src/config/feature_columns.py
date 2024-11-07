# src/config/feature_columns.py

from enum import Enum

class FeatureColumns(Enum):
    CATEGORICAL_COLUMNS = [
        'survey_type', 'province', 'gender', 'nationality', 'religion', 'religion_freq',
        'edu', 'political', 'health', 'health_problem', 'depression', 'hukou',
        'socialize', 'relax', 'learn', 'equity', 'class', 'work_exper',
        'work_status', 'work_type', 'work_manage', 'family_status', 'car', 'marital'
    ]
    NUMERICAL_COLUMNS = ['id', 'income', 'family_income']
    ONE_HOT_COLUMNS = ['province', 'nationality', 'hukou', 'work_exper', 'work_status', 'work_manage', 'marital', 'political']
    LOW_CORRELATION_FEATURES = ['survey_type', 'religion', 'work_status', 'work_type', 'work_manage']
