import datetime
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.metrics import r2_score
import analysis_function as ana


data = pd.read_csv("..\\..\\0. Data file\\Data-All.csv")
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:246, :9], data.iloc[:246, 15:20],
                                                    train_size=.80, random_state=13)

data_select = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 6, 7, 8]]

a = np.arange(1, 21, 4)  # 6+
b = np.arange(10, 310, 60)   # 50+
c = np.arange(10, 210, 40)   # 50+

# 分类数据，目标
for m in range(5):
    y_tra, y_tes = y_train.iloc[:, m:m+1],  y_test.iloc[:, m:m+1]
    y_tr = y_tra.values.ravel()
    y_te = y_tes.values.ravel()
    for n in range(3):
        k = data_select[n]
        x_tr, x_te = x_train.iloc[:, k], x_test.iloc[:, k]
        mean = x_tr.mean(axis=0)
        x_tr -= mean
        std = x_tr.std(axis=0)
        x_tr /= std
        x_te -= mean
        x_te /= std

        start_time = datetime.datetime.now()
        epoch = 0
        Epoch = a.size * b.size * c.size

        model_score = pd.DataFrame(columns=['max_features', 'n_estimators', 'max_depth', 'r2_val', 'r2'])
        key_name = ['max_features', 'n_estimators', 'max_depth', 'r2_val', 'r2']

        for i in a:
            for j in b:
                for k in c:
                    model_build = RandomForestRegressor(max_features=i, n_estimators=j, max_depth=k)
                    r2_val = cross_val_score(model_build, x_tr, y_tr, scoring='r2', cv=5).mean()

                    model = model_build.fit(x_tr, y_tr)
                    y_pred = model.predict(x_te)
                    r2 = r2_score(y_te, y_pred)

                    model_score = model_score.append({key_name[0]: i, key_name[1]: j,
                                                      key_name[2]: k, key_name[3]: r2_val, key_name[4]: r2}, ignore_index=True)
                    end_time = datetime.datetime.now()
                    running_time = end_time - start_time
                    epoch += 1
                    print('\rschedule:', epoch, '/', Epoch, ', runtime:', running_time, flush=True, end='')

        ana.analysis_parameter(model_score, key_name[0:3], key_name[3:5])

        model_score.to_csv('.\\score\\model_score_RF_{a}_{b}.csv'.format(a=m, b=n), columns=key_name)
