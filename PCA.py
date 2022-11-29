import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv('.\\0. Data file\\Data-All.csv')
data = data.iloc[:246, :9]
data_pca = data.apply(lambda x: (x-x.mean())/x.std())

pca = PCA(n_components=0.85)

pca.fit(data)
print(pca.explained_variance_ratio_)
