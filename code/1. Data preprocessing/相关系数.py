import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\MI\\Desktop\\BT邀稿\\Data-All-Pearson.csv")
data = data.iloc[:246, :13]

corr = data.corr()
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap="coolwarm")
plt.tight_layout()
plt.savefig('C:\\Users\\MI\\Desktop\\BT邀稿\\Fig\\{变量}-2.jpg'.format(变量='相关系数矩阵'), dpi=300)
plt.show()
