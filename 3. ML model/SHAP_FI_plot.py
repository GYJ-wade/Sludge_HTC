import matplotlib.pyplot as plt
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


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv("../0. Data file/Data-SHAP.csv")
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:246, :9], data.iloc[:246, 15:18],
                                                    train_size=.80, random_state=13)

name_title = ['HHV', 'CR', 'ER']
num_f = ['a', 'b', 'c', 'd', 'e', 'f']
data_select = [[0, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 6, 7, 8]]
drop_name = ['without C as input', 'without HHV as input']

plt.figure(figsize=(10, 6), dpi=300, facecolor='white')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
shap_total = np.zeros((6, 8))
for i in range(3):
    y_tr, y_te = y_train.iloc[:, i:i+1],  y_test.iloc[:, i:i+1]
    y_tr_1d = y_tr.values.ravel()
    for j in range(2):
        k = data_select[j]
        x_tr, x_te = x_train.iloc[:, k], x_test.iloc[:, k]
        model = RandomForestRegressor(max_features=3, n_estimators=100, max_depth=200).fit(x_tr, y_tr_1d)  # .to(device)
        y_pred = model.predict(x_te)
        rmse = np.sqrt(mean_squared_error(y_pred, y_te))
        r2 = r2_score(y_te, y_pred)
        result_pred(x_tr, y_tr, 'Train')
        result_pred(x_te, y_te, 'Test')

        plt.subplot(2, 3, j*3+i+1, facecolor='white')
        plt.title(label=name_title[i]+' ('+drop_name[j]+')')
        ax = plt.gca()
        plt.text(0.8, 0.1, '({})'.format(num_f[j+i*2]), horizontalalignment="left",
                 fontsize='10', transform=ax.transAxes)
        plt.grid(visible=True, color='silver', linewidth=1, axis="x")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_tr)
        shap_mean = np.zeros((2, 8))
        for k in range(8):
            shap_mean[0:1, k:k+1] = np.mean(np.abs(shap_values[:, k:k+1]))
        total = np.sum(shap_mean[0:1, :])
        for l in range(8):
            shap_mean[0:1, l:l+1] = shap_mean[0:1, l:l+1] / total * 2
            shap_total[i*2+j:i*2+j+1, l:l+1] = shap_mean[0:1, l:l+1] / 2
        shap.summary_plot(shap_mean, x_tr, show=False, plot_type="bar", plot_size=None, color='teal')
        plt.xlabel('Feature importance')
plt.tight_layout(pad=1.15)
plt.savefig('../5. Fig/SHAP-{name}.jpg'.format(name=' summary'), dpi=300)
print(shap_total)
plt.show()
