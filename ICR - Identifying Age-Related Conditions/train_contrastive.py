import json

from script.contrastive.model import run_tabular_experiment, evaluate_experiment_score

if __name__ == '__main__':
    with open('config.json') as config_file:
        config_project = json.load(config_file)
    
    config_project.update(
        {
            'NAME': 'contrastive_lgb',
            'NUM_SIMULATION': None,
            'LOG_EVALUATION': 250,
            'SAVE_MODEL': True,
            'INCREASE': True,
            'TRAIN_MODEL': True,
            'SAVE_RESULTS_PATH': 'experiment'
        }
    )
    print('Starting Experiment', config_project['NAME'])

    PARAMS_LGB = {
        'tree_learner': 'voting',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'n_jobs': -1,
        'num_leaves': 2**8,
        'learning_rate': 0.05,
        'feature_fraction': 0.75,
        'bagging_freq': 1,
        'bagging_fraction': 0.80,
        'lambda_l2': 1,
        'verbosity': -1,
        'n_round': 1500,
    }
    feature_list = config_project['ORIGINAL_FEATURE']
    
    if config_project['TRAIN_MODEL']:
        run_tabular_experiment(
            config_experiment=config_project, params_lgb=PARAMS_LGB, 
            feature_list=feature_list,
            num_simulation=config_project['NUM_SIMULATION'],
            target_col=config_project['TARGET_COL'], 
        )
    if config_project['SAVE_MODEL']:
        evaluate_experiment_score(
            config_experiment=config_project, params_lgb=PARAMS_LGB, 
            feature_list=feature_list,
            target_col=config_project['TARGET_COL']
        )