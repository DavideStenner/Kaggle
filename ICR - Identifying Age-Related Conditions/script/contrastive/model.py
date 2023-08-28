import os
import gc
import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

from typing import Tuple

from sklearn.metrics import roc_auc_score

from script.contrastive.augment import contrastive_pipeline, fe_pipeline, fe_new_col_name
from script.contrastive.loss import competition_log_loss

def get_dataset(
        data: pd.DataFrame, fold_: int, inference: bool,
        target_col: str, feature_list: list
    ) -> lgb.Dataset:
        
    mask_fold = data['fold']==fold_ if inference else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)
    train_x = data[feature_list].to_numpy('float32')
    train_y = data['target_contrast'].to_numpy('float32')

    lgb_dataset = lgb.Dataset(train_x, train_y)

    return lgb_dataset

def get_augment_dataset(
        data: pd.DataFrame, fold_: int, inference: bool,
        target_col: str, feature_list: list, num_simulation: int,
    ) -> lgb.Dataset:
    
    mask_fold = data['fold']==fold_ if inference else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)

    #tabaugment
    data = contrastive_pipeline(
        data=data, feature_list=feature_list, 
        original_tgt_label=target_col,
        num_simulation=num_simulation, inference=inference 
    )
    used_col = feature_list + fe_new_col_name()
    train_x = data[used_col].to_numpy('float32')
    train_y = data['target_contrast'].to_numpy('float32')

    lgb_dataset = lgb.Dataset(train_x, train_y)

    return lgb_dataset

def run_tabular_experiment(
        config_experiment: dict, params_lgb: dict,
        feature_list: list, num_simulation: int,
        target_col: str,
    ) -> None:

    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'], 
        config_experiment['NAME']
    )
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_list = []
    progress_list = []

    train = pd.read_pickle(
        os.path.join(config_experiment['PATH_DATA'], 'processed_data.pkl')
    )[feature_list + ['fold', target_col]]

    for fold_ in range(config_experiment['N_FOLD']):
        print(f'\n\nStarting fold {fold_}\n\n\n')
        
        train_matrix = get_augment_dataset(
            data=train, fold_=fold_, inference=False, 
            target_col=target_col, feature_list=feature_list,
            num_simulation=num_simulation,
        )
        test_matrix = get_augment_dataset(
            data=train, fold_=fold_, 
            inference=True, target_col=target_col, 
            feature_list=feature_list,
            num_simulation=num_simulation,
        )

        progress = {}

        callbacks_list = [
            lgb.record_evaluation(progress),
            lgb.log_evaluation(period=config_experiment['LOG_EVALUATION'], show_stdv=False)
        ]

        model = lgb.train(
            params=params_lgb,
            train_set=train_matrix, 
            num_boost_round=params_lgb['n_round'],
            valid_sets=[test_matrix],
            valid_names=['valid'],
            callbacks=callbacks_list,
        )

        if config_experiment['SAVE_MODEL']:
            model.save_model(
                os.path.join(
                       save_path,
                    f'lgb_{fold_}.txt'
                )
            )

        model_list.append(model)
        progress_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()
        if config_experiment['SAVE_MODEL']:
            save_lgb_model(
                model_list=model_list, progress_list=progress_list,
                save_path=save_path
            )

def save_lgb_model(
        model_list: list, progress_list: list, save_path: str
    )->None:
        with open(
            os.path.join(
                save_path,
                'model_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)

        with open(
            os.path.join(
                save_path,
                'progress_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)

def evaluate_experiment_score(
        config_experiment: dict, 
        params_lgb: dict, feature_list: list,
        target_col: dict
    ) -> None:

    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'], 
        config_experiment['NAME']
    )
    
    # Find best epoch
    with open(
        os.path.join(
            save_path,
            'progress_list_lgb.pkl'
        ), 'rb'
    ) as file:
        progress_list_lgb = pickle.load(file)

    with open(
        os.path.join(
            save_path,
            'model_list_lgb.pkl'
        ), 'rb'
    ) as file:
        model_list_lgb = pickle.load(file)

        
    progress_dict_lgb = {
        'time': range(params_lgb['n_round']),
    }

    progress_dict_lgb.update(
            {
                f"{params_lgb['metric']}_fold_{i}": progress_list_lgb[i]['valid'][params_lgb['metric']]
                for i in range(config_experiment['N_FOLD'])
            }
        )

    progress_df_lgb = pd.DataFrame(progress_dict_lgb)

    progress_df_lgb[f"average_{params_lgb['metric']}"] = progress_df_lgb.loc[
        :, [params_lgb['metric'] in x for x in progress_df_lgb.columns]
    ].mean(axis =1)
    
    progress_df_lgb[f"std_{params_lgb['metric']}"] = progress_df_lgb.loc[
        :, [params_lgb['metric'] in x for x in progress_df_lgb.columns]
    ].std(axis =1)

    best_epoch_lgb = (
        int(progress_df_lgb[f"average_{params_lgb['metric']}"].argmax())
        if config_experiment['INCREASE'] else
        int(progress_df_lgb[f"average_{params_lgb['metric']}"].argmin())
    )
    best_score_lgb = progress_df_lgb.loc[
        best_epoch_lgb,
        f"average_{params_lgb['metric']}"].max()
    lgb_std = progress_df_lgb.loc[
        best_epoch_lgb, f"std_{params_lgb['metric']}"
    ]

    print(f'Best epoch: {best_epoch_lgb}, CV-Auc: {best_score_lgb:.5f} ± {lgb_std:.5f}')

    best_result_lgb = {
        'best_epoch': best_epoch_lgb+1,
        'best_score': best_score_lgb
    }

    with open(
        os.path.join(
            save_path,
            'best_result_lgb.txt'
        ), 'w'
    ) as file:
        json.dump(best_result_lgb, file)

    get_retrieval_score(
        config_experiment=config_experiment,
        best_result_lgb=best_result_lgb, model_list_lgb=model_list_lgb,
        feature_list=feature_list, target_col=target_col
    )
    
    explain_model(config_experiment, best_result_lgb, model_list_lgb, feature_list)

def get_retrieval_score(
        config_experiment: dict,
        best_result_lgb: dict, model_list_lgb: Tuple[lgb.Booster, ...],
        feature_list: list, target_col: str
    ) -> None:
    
    data = pd.read_pickle(
        os.path.join(
            config_experiment['PATH_DATA'], 
            'processed_data.pkl'
        )
    )[feature_list + ['fold', target_col]]

    auc_ =  0
    comp_score = 0

    for fold_ in range(config_experiment['N_FOLD']):            
        test = data[data['fold']==fold_].reset_index(drop=True)
        
        #use for retrieval
        target_example_0 = data.loc[
            (data['fold']!=fold_) &
            (data[target_col] == 0), feature_list
        ].values

        target_example_1 = data.loc[
            (data['fold']!=fold_) &
            (data[target_col] == 1), feature_list
        ].values
        
        test_y = test[target_col].to_numpy('float32')

        retrieval_dataset_0 = get_retrieval_dataset(test, target_example_0, feature_list)
        retrieval_dataset_1 = get_retrieval_dataset(test, target_example_1, feature_list)

        used_feature = feature_list + fe_new_col_name()

        retrieval_dataset_0['pred'] = model_list_lgb[fold_].predict(
            retrieval_dataset_0[used_feature], 
            num_iteration = best_result_lgb['best_epoch']
        )
        pred_0 = retrieval_dataset_0.groupby('rows')['pred'].mean().reset_index().sort_values('rows')['pred'].values

        retrieval_dataset_1['pred'] = model_list_lgb[fold_].predict(
            retrieval_dataset_1[used_feature],
            num_iteration = best_result_lgb['best_epoch']
        )
        pred_1 = retrieval_dataset_1.groupby('rows')['pred'].mean().reset_index().sort_values('rows')['pred'].values
        
        pred_1 = pred_1/(pred_0+pred_1)

        auc_ += roc_auc_score(test_y, pred_1)/config_experiment['N_FOLD']
        comp_score += competition_log_loss(test_y, pred_1)/config_experiment['N_FOLD']

    print(f'Retrieval auc: {auc_:.5f}; Retrieval balanced log-loss: {comp_score:.5f}')

def get_retrieval_dataset(
        test: pd.DataFrame, target_example: pd.DataFrame, 
        feature_list:list
    ) -> Tuple[pd.DataFrame, list]:

    test_shape = test.shape[0]
    target_example_shape = target_example.shape[0]

    test_x = test[feature_list].to_numpy('float32')

    target_example = np.concatenate(
        [
            target_example
            for _ in range(test_shape)
        ], axis=0
    )
    test_x = np.repeat(test_x, target_example_shape, axis=0)
    index_test = np.repeat(test.index.values, target_example_shape, axis=0)

    retrieval_dataset = fe_pipeline(
        dataset_1=target_example,
        dataset_2=test_x, feature_list=feature_list,
    )
    retrieval_dataset['rows'] = index_test

    return retrieval_dataset

def explain_model(
        config_experiment: dict, best_result_lgb: dict, 
        model_list_lgb: Tuple[lgb.Booster, ...], feature_list: list,
    ) -> None:
    
    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'], 
        config_experiment['NAME']
    )
    
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = feature_list + fe_new_col_name()

    for fold_, model in enumerate(model_list_lgb):
        feature_importances[f'fold_{fold_}'] = model.feature_importance(
            importance_type='gain', iteration=best_result_lgb['best_epoch']
        )

    feature_importances['average'] = feature_importances[
        [f'fold_{fold_}' for fold_ in range(config_experiment['N_FOLD'])]
    ].mean(axis=1)

    fig = plt.figure(figsize=(12,8))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
    plt.title(f"50 TOP feature importance over {config_experiment['N_FOLD']} average")

    fig.savefig(
        os.path.join(save_path, 'importance_plot.png')
    )