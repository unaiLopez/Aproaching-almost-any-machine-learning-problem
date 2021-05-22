import numpy as np
from numpy.core.numeric import False_
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    data['kfold'] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    #calculate number of bins with Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(data))))
    print(num_bins)

    data['bins'] = pd.cut(
        data['target'], bins=num_bins, labels=False
    )

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    data = data.drop('bins', axis=1)

    return data

if __name__ == '__main__':
    X, y = datasets.make_regression(
        n_samples=15000, n_features=100, n_targets=1
    )

    df = pd.DataFrame(
        X,
        columns=[f'f_{i}' for i in range(X.shape[1])]
    )
    df['target'] = y
    df = create_folds(df)

    print(df.head())
    print(df.tail())