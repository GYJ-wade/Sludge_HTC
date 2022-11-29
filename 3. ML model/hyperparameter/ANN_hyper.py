import datetime
from keras import models, layers
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import analysis_function as ana
import tensorflow as tf
import keras

data = pd.read_csv("..\\..\\0. Data file\\Data-All.csv")
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:246, :9], data.iloc[:246, 15:20], train_size=.80,
                                                    random_state=13)

data_select = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 6, 7, 8]]

a = np.arange(40, 160, 40)
b = np.array([0, 0.1, 0.2, 0.3, 0.4])
c = np.arange(1, 6, 1)
d = np.array([0.001, 0.002, 0.003])


def model_ANN(x, RMSprop_rate):
    model = models.Sequential()
    activation = 'relu'
    model.add(layers.Dense(i, activation=activation, input_shape=[len(x.keys())]))
    model.add(layers.Dropout(j))
    model.add(layers.BatchNormalization())
    num = k
    while num > 1:
        model.add(layers.Dense(i, activation=activation))
        model.add(layers.Dropout(j))
        model.add(layers.BatchNormalization())
        num -= 1
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.RMSprop(RMSprop_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

EPOCHS = 2000


for m in range(5):
    y_tra, y_tes = y_train.iloc[:, m:m + 1], y_test.iloc[:, m:m + 1]
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
        Epoch = a.size * b.size * c.size * d.size

        model_score = pd.DataFrame(columns=['dense', 'dropout', 'layer', 'rate', 'r2_tr', 'r2_val', 'r2_te'])
        key_name = ['dense', 'dropout', 'layer', 'rate', 'r2_tr', 'r2_val', 'r2_te']

        for i in a:
            for j in b:
                for k in c:
                    for l in d:
                        r2_tr_num = np.zeros(5)
                        r2_val_num = np.zeros(5)
                        for k_fold_num in range(5):
                            num_val_samples = len(x_tr) // 5

                            x_tr_k = x_tr[k_fold_num * num_val_samples: (k_fold_num + 1) * num_val_samples]
                            y_tr_k = y_tr[k_fold_num * num_val_samples: (k_fold_num + 1) * num_val_samples]

                            x_val_k = np.concatenate([x_tr[:k_fold_num * num_val_samples], x_tr[(k_fold_num + 1) * num_val_samples:]], axis=0)
                            y_val_k = np.concatenate([y_tr[:k_fold_num * num_val_samples], y_tr[(k_fold_num + 1) * num_val_samples:]], axis=0)

                            model = model_ANN(x_tr_k, l)
                            model.fit(x_tr_k, y_tr_k, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=early_stop)

                            y_pred_tr = model.predict(x_tr_k)
                            r2_tr_k = r2_score(y_tr_k, y_pred_tr)
                            r2_tr_num[k_fold_num] = r2_tr_k

                            y_pred_val = model.predict(x_val_k)
                            r2_val_k = r2_score(y_val_k, y_pred_val)
                            r2_val_num[k_fold_num] = r2_val_k

                        r2_tr = np.mean(r2_tr_num)
                        r2_val = np.mean(r2_val_num)

                        model = model_ANN(x_tr, l)
                        model.fit(x_tr, y_tr, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=early_stop)
                        y_pred = model.predict(x_te)
                        r2_te = r2_score(y_te, y_pred)

                        model_score = model_score.append({key_name[0]: i, key_name[1]: j, key_name[2]: k,
                                                          key_name[3]: l, key_name[4]: r2_tr, key_name[5]: r2_val,
                                                          key_name[6]: r2_te},
                                                         ignore_index=True)
                        end_time = datetime.datetime.now()
                        running_time = end_time - start_time
                        epoch += 1
                        print('\rschedule:', epoch, '/', Epoch, ', runtime:', running_time, flush=True, end='')

        ana.analysis_parameter(model_score, key_name[0:4], key_name[4:7])

        model_score.to_csv('.\\score\\model_score_ANN_{a}_{b}.csv'.format(a=m, b=n), columns=key_name)
