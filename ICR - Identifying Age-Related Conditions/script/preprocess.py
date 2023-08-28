import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

def preprocess_data(config: dict) -> None:

    greeks = pd.read_csv(
        os.path.join(
        config['PATH_DATA_ORIGINAL'],
        'greeks.csv'
        )
    )
    train = pd.read_csv(
        os.path.join(
        config['PATH_DATA_ORIGINAL'],
        'train.csv'
        )
    )
    greeks['Epsilon'] = pd.to_datetime(
        greeks['Epsilon'].replace('Unknown', np.nan)
    )

    fold_info = pd.merge(
        train[['Id', 'Class']], 
        greeks, on='Id'
    )
    split = StratifiedKFold(n_splits=config['N_FOLD'])
    iterator_split = enumerate(split.split(fold_info, fold_info['Alpha']))
    fold_info['fold'] = int(-1)

    for fold_, (_, test_index) in iterator_split:
        fold_info.loc[test_index, 'fold'] = int(fold_)

    assert (fold_info['fold'] == -1).sum() == 0
    assert fold_info['fold'].nunique() == config['N_FOLD']

    train = pd.merge(
        train,
        fold_info[
            [
                'Id', 'Alpha', 'Beta', 
                'Gamma', 'Delta', 'Epsilon',
                'fold'
            ]
        ], on='Id'
    )
    train['EJ'] = train['EJ'].map(
        {
            'A': 0,
            'B': 1
        }
    ).astype('uint8')

    train.to_pickle(
        os.path.join(
            config['PATH_DATA'],
            'processed_data.pkl'
        )
    )