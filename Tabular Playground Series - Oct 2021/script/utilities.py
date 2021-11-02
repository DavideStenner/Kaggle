import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

#CONSTANT
RANDOM_STATE = 383920

TARGET_COL = 'target'
CAT_TRESHOLD = 32

N_FOLD = 5
FOLD_STRAT_NAME = 'FOLD'
REDUCED_FOLD_NAME = FOLD_STRAT_NAME + '-REDUCED'


###LGB
PARAMS_LGB_BASE = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': -1,
        'num_leaves': 2**8,
        'n_jobs':-1,
        'verbose': -1,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'random_state': RANDOM_STATE
}

###XGB
PARAMS_XGB_GPU_BASE = {
    'max_depth': 6,
    'learning_rate': 1e-3,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
}

PARAMS_XGB_BASE = {
    'max_depth': 6,
    'learning_rate': 1e-3,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'tree_method': 'hist',
}

###NN
EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = True

#############################
#############################
#FUNCTION
def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):
    """
    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
    avg_loss_limit - same but calculates avg throughout the series.
    na_loss_limit - not really useful.
    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
    """
    is_float = str(col.dtypes)[:5] == 'float'
    na_count = col.isna().sum()
    n_uniq = col.nunique(dropna=False)
    try_types = ['float16', 'float32']

    if na_count <= na_loss_limit:
        try_types = ['int8', 'int16', 'float16', 'int32', 'float32']

    for type in try_types:
        col_tmp = col

        # float to int conversion => try to round to minimize casting error
        if is_float and (str(type)[:3] == 'int'):
            col_tmp = col_tmp.copy().fillna(fillna).round()

        col_tmp = col_tmp.astype(type)
        max_loss = (col_tmp - col).abs().max()
        avg_loss = (col_tmp - col).abs().mean()
        na_loss = np.abs(na_count - col_tmp.isna().sum())
        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

        if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:
            return col_tmp

    # field can't be converted
    return col

def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):
    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes

        # collect stats
        na_count = df[col].isna().sum()
        n_uniq = df[col].nunique(dropna=False)
        
        # numerics
        if col_type in numerics:
            df[col] = sd(df[col])

        # strings
        if (col_type == 'object') and obj_to_cat:
            df[col] = df[col].astype('category')
        
        if verbose:
            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')
        new_na_count = df[col].isna().sum()
        if (na_count != new_na_count):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')
        new_n_uniq = df[col].nunique(dropna=False)
        if (n_uniq != new_n_uniq):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')

    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df




def gmean_ensembler(df,cols,plot=True):
    df = df[cols]
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    if plot:
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(len(cols)+2, len(cols)+2))

        # Draw the heatmap with the mask and correct aspect ratio
        _ = sns.heatmap(corr,mask=mask,cmap='prism',center=0, linewidths=1,
                        annot=True,fmt='.4f', cbar_kws={"shrink":.2})
        plt.show()
    rank = np.tril(corr.values,-1)
    rank[rank<0.7] = 1
    m = (rank>0).sum() - (rank==1).sum()
    m_gmean, s = 0, 0

    for n in range(m):
        mx = np.unravel_index(rank.argmin(), rank.shape)
        w = (m-n)/m
        m_gmean += w*(np.log1p(df.iloc[:,mx[0]])+np.log1p(df.iloc[:,mx[1]]))/2
        s += w
        rank[mx] = 1
    m_gmean = np.expm1(m_gmean/s)
    return(m_gmean)

