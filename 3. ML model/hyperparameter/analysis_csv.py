import analysis_function as ana
import pandas as pd

pred_type = ['HHV', 'CR', 'ER', 'CR-', 'ER-']
data_type = ['all', 'part 1', 'part 2']

key_name_GBT = ['learning_rate', 'max_iter', 'max_depth', 'r2_val', 'r2']
for m in range(5):
    for n in range(3):
        model_score = pd.read_csv('.\\score\\model_score_GBT_{a}_{b}.csv'.format(a=m, b=n))
        print('\nGBT, target:{a}, data:{b}'.format(a=pred_type[m], b=data_type[n]))
        ana.analysis_parameter(model_score, key_name_GBT[0:3], key_name_GBT[3:5])

key_name_RF = ['max_features', 'n_estimators', 'max_depth', 'r2_val', 'r2']
for m in range(5):
    for n in range(3):
        model_score = pd.read_csv('.\\score\\model_score_RF_{a}_{b}.csv'.format(a=m, b=n))
        print('\nRF, target:{a}, data:{b}'.format(a=pred_type[m], b=data_type[n]))
        ana.analysis_parameter(model_score, key_name_RF[0:3], key_name_RF[3:5])

key_name_ANN = ['dense', 'dropout', 'layer', 'rate', 'tr_r2', 'te_r2']
for m in range(5):
    for n in range(3):
        model_score = pd.read_csv('.\\score\\model_score_ANN_{a}_{b}.csv'.format(a=m, b=n))
        print('\nANN, target:{a}, data:{b}'.format(a=pred_type[m], b=data_type[n]))
        ana.analysis_parameter(model_score, key_name_ANN[0:4], key_name_ANN[4:6])
