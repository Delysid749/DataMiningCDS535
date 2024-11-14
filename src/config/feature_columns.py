from enum import Enum


class FeatureColumns(Enum):
    CATEGORICAL_COLUMNS = [
        'survey_type', 'province', 'gender', 'nationality', 'religion', 'religion_freq',
        'edu', 'political', 'health', 'health_problem', 'depression', 'hukou',
        'socialize', 'relax', 'learn', 'equity', 'class', 'work_exper',
        'work_status', 'work_type', 'work_manage', 'family_status', 'car', 'marital'
    ]
    NUMERICAL_COLUMNS = ['id', 'income', 'family_income']
    ONE_HOT_COLUMNS = ['province', 'hukou', 'work_exper', 'marital', 'political']
    LOW_CORRELATION_FEATURES = ['work_manage', 'work_status', 'nationality', 'religion_freq', 'gender', 'work_type',
                                'survey_type', 'religion']
