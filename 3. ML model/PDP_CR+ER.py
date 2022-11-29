import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from scipy.interpolate import splev, splrep


def result_pred(x, y, name):
    y_pred = model.predict(x)
    rmse = np.sqrt(mean_squared_error(y_pred, y))
    r2 = r2_score(y, y_pred)
    print('{}, rmse:{:5.4f}, r2:{:5.4f}'.format(name, rmse, r2))


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv("../0. Data file/Data-All.csv")
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:246, :9], data.iloc[:246, 16:18],
                                                    train_size=.80, random_state=13)

name_title = ['CR', 'ER']
unit_title = ['(%)', '(%)']
name_label = ['Moisture', 'Temperature', 'Time', 'N/C(O)', 'H/C(O)', 'HHV(O)', 'C(O)']
name_label_2 = ['Moisture', 'Temperature', 'RT', 'N/C', 'H/C', 'HHV', 'C']
name_label_1 = ['Moisture', 'Temperature', 'Time', 'NC(O)', 'HC(O)', 'HHV(O)', 'C(O)']
unit_label = ['(%)', '(°C)', '(min)', '', '', '(MJ/kg)', '(%)']
num = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
file_name = ['PDP_CR', 'PDP_ER']

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '15'

for i in range(2):
    y_tr, y_te = y_train.iloc[:, i:i+1], y_test.iloc[:, i:i+1]
    y_tr_1d = y_tr.values.ravel()
    model = RandomForestRegressor(max_features=18, n_estimators=300, max_depth=160).fit(x_train, y_tr_1d)
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_te))
    r2 = r2_score(y_te, y_pred)
    result_pred(x_train, y_tr, 'Train')
    result_pred(x_test, y_te, 'Test')
    for j in range(7):
        plt.figure(figsize=(4.5, 4), dpi=300, facecolor='white')
        plt.grid(visible=True, color='silver', linewidth=1)

        sns.set_theme(style="ticks", palette="deep", font_scale=1.3)

        pdp = partial_dependence(model, x_train, [name_label[j]],
                                 kind="average",
                                 method='brute',
                                 grid_resolution=50)
        plot_x = pdp['values'][0]
        plot_y = pdp['average'][0]
        tck = splrep(plot_x, plot_y, s=30)
        xnew = np.linspace(plot_x.min(), plot_x.max(), 300)
        ynew = splev(xnew, tck, der=0)

        plt.plot(plot_x, plot_y, color='orangered', alpha=0.6, linewidth=4)

        sns.rugplot(data=x_train, x=name_label[j], height=.06, color='r', alpha=0.3, linewidth=4)

        x_min = plot_x.min() - (plot_x.max() - plot_x.min()) * 0.1
        x_max = plot_x.max() + (plot_x.max() - plot_x.min()) * 0.1
        plt.xlabel(name_label_2[j] + ' ' + unit_label[j])
        plt.ylabel('Partial dependence')
        plt.xlim(x_min, x_max)
        ax = plt.gca()
        plt.text(0.05, 0.9, '({})'.format(num[j]), horizontalalignment="left", transform=ax.transAxes)
        plt.subplots_adjust(left=0.2, bottom=0.17, right=0.92, top=0.88, wspace=None,)
        plt.savefig('../5. Fig/{file}/PDP-{name}-{feature}.jpg'.format(file=file_name[i], name=name_title[i], feature=name_label_1[j]), dpi=300)  # name_title[i]
