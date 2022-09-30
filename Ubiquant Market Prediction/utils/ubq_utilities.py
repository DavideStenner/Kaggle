import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple
from scipy.stats import spearmanr
import scipy
from tensorflow.keras import mixed_precision
import tensorflow as tf

#CONSTANT
RANDOM_STATE = 385619

SUBSAMPLE = 2
N_FOLD = 5
EMBARGO = 50
MIN_TIME_TO_USE = 0

TIME_COL = 'time_id'
FOLD_NAME = 'fold_cv'
TARGET_COL = 'target'
SUBSAMPLE_FOLD = 'subsample_fold'

STARTING_NUMERIC_FEAT_NAME = [f'f_{x}' for x in range(300)]
STARTING_CAT_FEAT_NAME = ['investment_id']
STARTING_FEATURE_NAME = STARTING_CAT_FEAT_NAME + STARTING_NUMERIC_FEAT_NAME

## FEATURE EXPOSURE
#https://github.com/numerai/example-scripts/tree/495d3c13153c2068a87cf8c33196a787b2a0871f
def feature_exposures(df, pred_col):
    feature_names = [
        f for f in df.columns if f in STARTING_NUMERIC_FEAT_NAME 
    ]
    
    exposures = []
    
    for f in feature_names:
        fe = spearmanr(df[pred_col], df[f])[0]
        exposures.append(fe)
    return np.array(exposures)


def max_feature_exposure(df, pred_col):
    return np.max(np.abs(feature_exposures(df, pred_col)))


def feature_exposure(df, pred_col):
    return np.sqrt(np.mean(np.square(feature_exposures(df, pred_col))))


# FEATURE NEUTRALIZATION
def neutralize(df,
               columns,
               neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era",
               progress=True
              ):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        if progress:
            print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)

#CORR CALCULATION

#####XGB

def calculate_corr(df: pd.DataFrame, time_col: str='time_id', pred_col: str='y_pred', target_col: str='y_true', sharpe=False):
    df_used = df[[time_col, pred_col, target_col]]
    corr_by_time_id = df.groupby(time_col).apply(
        lambda x: x[[pred_col]].corrwith(x[target_col])
    )
    
    mean_corr = corr_by_time_id.mean().values
    if sharpe:
        std_corr = corr_by_time_id.std().values
        return mean_corr/std_corr
    else:
        return mean_corr
    
def corr_sharpe_lgb(
    time_id_fold,
    y_pred: np.array, dtrain: lgb.Dataset, 
) -> Tuple[str, float, bool]:
    """
    Pearson correlation coefficient metric
    """
    y_true = dtrain.get_label()
    
    pd_info = pd.DataFrame(
        {
            'time_id': time_id_fold,
            'y_pred': y_pred,
            'y_true': y_true
        }
    )
    sharpe_corr = calculate_corr(pd_info, sharpe=True)[0]
    return 'pearson_corr_sharpe', sharpe_corr, True

def corr_sharpe_xgb(
    time_id_fold,
    y_pred: np.array, dtrain: xgb.DMatrix, 
) -> Tuple[str, float]:
    """
    Pearson correlation coefficient metric
    """
    y_true = dtrain.get_label()
    
    pd_info = pd.DataFrame(
        {
            'time_id': time_id_fold,
            'y_pred': y_pred,
            'y_true': y_true
        }
    )
    sharpe_corr = calculate_corr(pd_info, sharpe=True)
    
    return 'pearson_corr_sharpe', sharpe_corr


def corr_xgb(
    time_id_fold,
    y_pred: np.array, dtrain: xgb.DMatrix, 
) -> Tuple[str, float]:
    """
    Pearson correlation coefficient metric
    """
    y_true = dtrain.get_label()
    
    pd_info = pd.DataFrame(
        {
            'time_id': time_id_fold,
            'y_pred': y_pred,
            'y_true': y_true
        }
    )
    mean_corr = calculate_corr(pd_info)
    
    return 'pearson_corr', mean_corr


def get_time_series_cross_val_splits(
    data, time_col = TIME_COL, cv = N_FOLD, 
    embargo = EMBARGO, min_time_to_use = 0,
    percent_split=False
):
    #https://github.com/numerai/example-scripts/blob/495d3c13153c2068a87cf8c33196a787b2a0871f/utils.py#L79
    
    all_train_eras = data[time_col].unique()    
    
    if percent_split:
        min_eras = all_train_eras.min()
        max_eras = all_train_eras.max()

        if min_time_to_use > 0:
            print('Min time to use not implemented for percent split')
            
        cumulative_size = data[[time_col]].sort_values(time_col).groupby(time_col).size()

        cum_size_time = cumulative_size.cumsum()/cumulative_size.sum()

        percent_split = 1/cv
        time_split = [cum_size_time[cum_size_time >= percent_split*i].index[0] for i in range(cv)]
        max_eras = data[time_col].max()

        test_splits = [
            all_train_eras[
                (all_train_eras >= (time_split[i] if i > 0 else min_eras)) &
                (all_train_eras <= (time_split[i+1] if i < (cv - 1) else max_eras))
            ] for i in range(cv)
        ]
    
    else:
        number_eras = len(all_train_eras)
       
        #each test split has this length
        len_split = (
            number_eras - min_time_to_use
        ) // cv

        #create kfold split by selecting also a min time to use --> first min_time_to_use won't be use for test split
        #fix the last test split to have all the last eras, in case the number of eras wasn't divisible by cv
        test_splits = [
            all_train_eras[
                (min_time_to_use + (i * len_split)):
            (min_time_to_use + (i + 1) * len_split) if i < (cv - 1) else number_eras
            ] for i in range(cv)
        ]

    train_splits = []
    for test_split in test_splits:
        
        #get boundaries
        test_split_min, test_split_max = int(np.min(test_split)), int(np.max(test_split))

        # get all of the eras that aren't in the test split
        train_split_not_embargoed = [e for e in all_train_eras if not (test_split_min <= int(e) <= test_split_max)]
        
        # embargo the train split so we have no leakage.
        train_split = [
            e for e in train_split_not_embargoed if
            abs(int(e) - test_split_max) > embargo and abs(int(e) - test_split_min) > embargo
        ]
        train_splits.append(train_split)

    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip

## SAFE MEMORY REDUCTION
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
    for col in df.columns:
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


# Function to get hardware strategy
def get_hardware_strategy(mixed_f16=False):
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        if mixed_f16:
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_global_policy(policy)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()
        if mixed_f16:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
        
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return tpu, strategy

