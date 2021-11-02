import pandas as pd
import lightgbm as lgb
import time
from utilities import (
    TARGET_COL, REDUCED_FOLD_NAME
)


def train_feat_sel(data, feature_list, feat_del_list, target_name, fold_strat_name,
                   params, cat_col, verbose_eval, early_stopping_rounds):
    """
    Function which make one step of validation and evaluate. feat_del_list is taken out from feature_list to train the model (lgb)
    fold_strat_name == 0 is train; == 1 is test
    """
    score = 0
    
    feature_list = [x for x in feature_list if x not in feat_del_list]
    cat_col = [x for x in cat_col if x not in feat_del_list]
    
    mask_train = (data[fold_strat_name] == 0)
    mask_test = (data[fold_strat_name] == 1)


    train_x, train_y = data.loc[mask_train, feature_list], data.loc[mask_train, target_name]
    test_x, test_y = data.loc[mask_test, feature_list], data.loc[mask_test, target_name]
    
    model = lgb.train(
        params,
        lgb.Dataset(train_x, label=train_y,categorical_feature=cat_col), 10000,
        valid_sets = lgb.Dataset(test_x, label=test_y,categorical_feature=cat_col),
        valid_names ='validation', verbose_eval= verbose_eval, early_stopping_rounds = early_stopping_rounds,
    )

    #evaluate score and save model for importance/prediction
    score = model.best_score['validation']['auc']
    
    return score


def feat_selection_pipeline(data, feature_list, possible_del_list,  
                            params, cat_col, target_name = TARGET_COL, fold_strat_name = REDUCED_FOLD_NAME, 
                            new_learning_rate = -1, verbose_eval = -1, early_stopping_rounds = 10):
    """
    iterate train_feat_sel over each variable
    """
    print('\n\n')
    #overwrite learning rate --> fast iteration
    params['learning_rate'] = new_learning_rate if (new_learning_rate != -1) else params['learning_rate']

    params['verbose'] = -1
    
    del_list = []
    starting_time = time.time()

    score = train_feat_sel(data = data, feature_list = feature_list, feat_del_list = del_list, target_name = target_name, 
                           params = params, cat_col = cat_col, fold_strat_name = fold_strat_name, 
                           verbose_eval = verbose_eval, early_stopping_rounds = early_stopping_rounds)
    print(f'Base score: {score:.6f}\n\n\n')
    min_one_iteration = (time.time()-starting_time)/60
    
    print(f'Minute one iteration: {min_one_iteration:.1f}; All iteration: {(min_one_iteration * len(possible_del_list)):.1f}\n\n\n')
    
    starting_score = score
    
    for feat in possible_del_list:
        
        possible_del_list = del_list + [feat]
        score_feature = train_feat_sel(data = data, feature_list = feature_list, feat_del_list = possible_del_list, target_name = target_name, 
                           params = params, cat_col = cat_col, fold_strat_name = fold_strat_name, 
                           verbose_eval = verbose_eval, early_stopping_rounds = early_stopping_rounds)
        
        if score_feature > score:
            print(f'\n\nFeature {feat} dropped. Delta AUC: {(score_feature - score):.6f}\n\n')
            
            score = score_feature
            del_list.append(feat)
            
    print(f'Delta AUC: {score - starting_score}; Number of feature dropped: {len(del_list)}\n\n')
    
    return del_list
