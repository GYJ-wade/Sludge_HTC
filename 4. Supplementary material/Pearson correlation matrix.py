import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("..\\0. Data file\\Data-Pearson.csv")
data = data.iloc[:246, :13]

corr = data.corr()
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap="coolwarm")
plt.tight_layout()
plt.savefig('..\\5. Fig\\{name}.jpg'.format(name='Pearson correlation matrix'), dpi=300)
plt.show()
