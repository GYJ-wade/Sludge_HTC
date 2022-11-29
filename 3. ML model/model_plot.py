import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras import models, layers
import tensorflow as tf
import keras
import datetime


def result_pred(x, y, name):
    y_pred = model.predict(x)
    rmse = np.sqrt(mean_squared_error(y_pred, y))
    r2 = r2_score(y, y_pred)
    print('{}, rmse:{:5.4f}, r2:{:5.4f}'.format(name, rmse, r2))


def plot_Test_Pred(x, y, r2, rmse, num, unit, title, major):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    test_predictions = model.predict(x)
    ax = plt.gca()
    plt.grid(visible=True, color='silver', linewidth=1)
    plt.scatter(y, test_predictions, s=300, c='teal', alpha=0.7, marker='o')
    sns.regplot(x=y.values.ravel(), y=test_predictions, ci=95, scatter=False, color='r', truncate=False)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('Experimental Values (' + unit + ')', fontsize='35', labelpad=10)
    plt.ylabel('Predictive Values (' + unit + ')', fontsize='35', labelpad=10)
    x_major_locator = MultipleLocator(major)
    y_major_locator = MultipleLocator(major)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.axis('equal')
    plt.axis('square')
    _ = plt.plot([-10, 600], [-10, 600], color='grey')
    plt.text(0.95, 0.07, title + '\nR$\mathregular{^2}$' + ': {:4.3f}\nRMSE: {:4.3f}'.format(r2, rmse),
             horizontalalignment="right",
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.9), fontsize='30',
             transform=ax.transAxes)
    plt.text(0.05, 0.9, '({})'.format(num), horizontalalignment="left",
             fontsize='30', transform=ax.transAxes)


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        schedule = 100 * epoch / EPOCHS
        end_time = datetime.datetime.now()
        running_time = end_time - start_time
        print('\rschedule: %.6f' % schedule, '%', ', epoch: ', epoch, '/', EPOCHS, ', runtime:', running_time,
              flush=True, end='')


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
data = pd.read_csv("../0. Data file/Data-All.csv")
x_tr, x_te, y_train, y_test = train_test_split(data.iloc[:246, :9], data.iloc[:246, 15:20],
                                               train_size=.80, random_state=13)

plt.figure(figsize=(24, 24), dpi=300)
num_f = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
unit = ['MJ/kg', '%', '%']
name_title = ['HHV', 'CR', 'ER']
name_model = ['GBT', 'RF', 'ANN']
num_GBT = [[0.6, 160, 100], [0.35, 160, 100], [0.2, 160, 100]]
num_RF = [[5, 70, 170], [1, 130, 170], [5, 70, 170]]
num_ANN = [[120, 0.2, 1, 0.002], [90, 0.2, 2, 0.001], [120, 0, 4, 0.001]]
major_base = [10, 20, 20]

for i in range(9):
    j = i % 3
    k = i // 3
    if k == 0:
        y_tr, y_te = y_train.iloc[:, 0:1], y_test.iloc[:, 0:1]
    elif k == 1:
        y_tr, y_te = y_train.iloc[:, 1:2], y_test.iloc[:, 1:2]
    elif k == 2:
        y_tr, y_te = y_train.iloc[:, 2:3], y_test.iloc[:, 2:3]
    y_tr_1d = y_tr.values.ravel()
    mean = x_tr.mean(axis=0)
    x_tr -= mean
    std = x_tr.std(axis=0)
    x_tr /= std
    x_te -= mean
    x_te /= std
    if j == 0:
        model_build = HistGradientBoostingRegressor(learning_rate=num_GBT[k][0], max_depth=num_GBT[k][1],
                                                    max_iter=num_GBT[k][2], n_iter_no_change=30)
        model = model_build.fit(x_tr, y_tr_1d)
        y_pr = model.predict(x_te)
        rmse = np.sqrt(mean_squared_error(y_pr, y_te))
        r2 = r2_score(y_te, y_pr)
    elif j == 1:
        model_O = RandomForestRegressor
        model = model_O(max_features=num_RF[k][0], n_estimators=num_RF[k][1], max_depth=num_RF[k][2]).fit(x_tr, y_tr_1d)
        y_pr = model.predict(x_te)
        rmse = np.sqrt(mean_squared_error(y_pr, y_te))
        r2 = r2_score(y_te, y_pr)
    elif j == 2:
        model = models.Sequential()
        activation = 'relu'
        model.add(layers.Dense(num_ANN[k][0], activation=activation, input_shape=[len(x_tr.keys())]))
        model.add(layers.Dropout(num_ANN[k][1]))
        model.add(layers.BatchNormalization())
        num = num_ANN[k][2]
        while num > 1:
            model.add(layers.Dense(num_ANN[k][0], activation=activation))
            model.add(layers.Dropout(num_ANN[k][1]))
            model.add(layers.BatchNormalization())
            num -= 1
        model.add(layers.Dense(1))
        optimizer = tf.keras.optimizers.RMSprop(num_ANN[k][3])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

        EPOCHS = 2000
        start_time = datetime.datetime.now()
        history = model.fit(x_tr, y_tr, epochs=EPOCHS,
                            validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

        print('\n')

        loss, mae, mse = model.evaluate(x_te, y_te, verbose=2)
        print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
        y_pr = model.predict(x_te)
        rmse = np.sqrt(mean_squared_error(y_pr, y_te))
        r2 = r2_score(y_te, y_pr)

    print()
    result_pred(x_tr, y_tr, 'Train')
    result_pred(x_te, y_te, 'Test')

    plt.subplot(3, 3, i + 1)
    plot_Test_Pred(x_te, y_te, r2, rmse, num_f[i], unit[k], name_model[j] + ' for ' + name_title[k], major_base[k])

plt.tight_layout(pad=1.3)
plt.savefig('../5. Fig/model_plot-{name}.jpg'.format(name='9 in 1'), dpi=300)
