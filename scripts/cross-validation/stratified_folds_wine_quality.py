import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('../../datasets/train_winequality-red.csv')

    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.quality.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

    df.to_csv('../../datasets/train_folds_winequality-red.csv')