import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import shap


def result_pred(x, y, name):
    y_pred = model.predict(x)
    rmse = np.sqrt(mean_squared_error(y_pred, y))
    r2 = r2_score(y, y_pred)
    print('{}, rmse:{:5.4f}, r2:{:5.4f}'.format(name, rmse, r2))


def plot_Test_Pred(x, y, r2, rmse):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    test_predictions = model.predict(x)
    plt.figure(figsize=(9, 9), dpi=300)
    plt.subplot(facecolor="darkslategray")
    plt.grid(visible=True, color='white', linewidth=1)  # 设置参考线
    plt.title(label=os.path.basename(__file__), loc='center', fontsize='20')
    plt.scatter(y, test_predictions, s=300, c='gold', alpha=0.7, marker='o')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('True Values (MJ/kg)', fontsize='25', labelpad=10)
    plt.ylabel('Predictions (MJ/kg)', fontsize='25', labelpad=10)
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(0, )
    plt.ylim(0, )
    _ = plt.plot([-10, 500], [-10, 500], color='lime')
    plt.text(test_predictions.max(), 2.5, 'r2: {:5.4f}\nrmse: {:5.4f}'.format(r2, rmse), horizontalalignment="right",
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.9), fontsize='20')


def plot_compar(x, y_test):
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = '20'
    y_pred = model.predict(x)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(len(y_pred)), y_test, color='blue', label="Predictions (MJ/kg)", linewidth=3, ls="dotted")
    plt.plot(range(len(y_pred)), y_pred, color='red', label="True Values (MJ/kg)", linewidth=3, ls="dotted")
    plt.title(label='RF', fontdict={'color': 'k', 'size': 30}, loc='center', pad=20)
    plt.legend(loc="upper right")

    ax = plt.gca()
    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)

    plt.xlabel('HHV', fontdict={'color': 'k', 'size': 25}, labelpad=10)
    plt.ylabel('Values', fontdict={'color': 'k', 'size': 25}, labelpad=10)
    plt.grid(visible=True, color='gray', linewidth=1)

    plt.gcf().subplots_adjust(left=None, top=None, bottom=0.15, right=None)


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv("..\\0. Data file\\Data-SHAP.csv")
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:246, :9], data.iloc[:246, 15:18],
                                                    train_size=.80, random_state=13)

name_title = ['HHV', 'CR', 'ER']
num_f = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
data_select = [[0, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 6, 7, 8]]
drop_name = ['without C as input', 'without HHV as input']

plt.figure(figsize=(12, 10), dpi=300, facecolor='white')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '12'

for i in range(3):
    y_tr, y_te = y_train.iloc[:, i:i+1],  y_test.iloc[:, i:i+1]
    y_tr_1d = y_tr.values.ravel()
    for j in range(2):
        k = data_select[j]
        x_tr, x_te = x_train.iloc[:, k], x_test.iloc[:, k]

        model = RandomForestRegressor(max_features=18, n_estimators=300, max_depth=160).fit(x_tr, y_tr_1d)
        y_pred = model.predict(x_te)
        rmse = np.sqrt(mean_squared_error(y_pred, y_te))
        r2 = r2_score(y_te, y_pred)
        result_pred(x_tr, y_tr, 'Train')
        result_pred(x_te, y_te, 'Test')

        plt.subplot(2, 3, j*3+i+1, facecolor='white')
        plt.title(label=name_title[i]+' ('+drop_name[j]+')')
        ax = plt.gca()
        plt.text(0.1, 0.1, '({})'.format(num_f[j+i*2]), horizontalalignment="left",
                 bbox=dict(boxstyle='round, pad=0.5', fc='white', ec='white', lw=1, alpha=0.9), fontsize='12', transform=ax.transAxes)
        plt.grid(visible=True, color='silver', linewidth=1)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_tr)
        shap.summary_plot(shap_values, x_tr, show=False, plot_size=None)
        plt.xlabel('SHAP value')
plt.tight_layout(pad=1.15)
plt.savefig('..\\5. Fig\\SHAP-{name}.jpg'.format(name='SHAP value'), dpi=300)  # name_title[i]
plt.show()
