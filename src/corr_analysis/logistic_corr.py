from main_corr import analyze_happiness_correlation

# 设置文件路径
test_file = '../../data/happiness_test.csv'
prediction_results_file = '../../report/logistic_regression/prediction_results.csv'
output_image_file = '../../report/logistic_regression/happiness_correlation.png'

analyze_happiness_correlation(test_file, prediction_results_file, output_image_file)