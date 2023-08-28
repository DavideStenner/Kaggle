import numpy as np
import pandas as pd

from typing import Tuple
from itertools import combinations

def get_all_combination(data: pd.DataFrame, inference: bool) -> Tuple[list, list]:
    index_number = list(range(data.shape[0]))
    list_all_combination = list(
        combinations(
            index_number,
            2
        )
    )
    #tabaugment only not missing values
    if num_simulation is None:
        num_simulation = len(list_all_combination)
        if not inference:
            print(f'Applying contrastive training')
            print(f'Using maximum number of combination: {num_simulation}')

    c_1_simulated = [c_1 for c_1, _ in list_all_combination[:num_simulation]]
    c_2_simulated = [c_2 for _, c_2 in list_all_combination[:num_simulation]]
    return c_1_simulated, c_2_simulated

def get_stratified_example(
        data: pd.DataFrame, original_tgt_label: str, inference: bool,
    ) -> Tuple[list, list]:

    c_1_simulated = []
    c_2_simulated = []
    target_sample = {
        0: (4 if inference else 8),
        1: (20 if inference else 40)
    }
    for row in range(data.shape[0]):
        curr_target = data.loc[row, original_tgt_label]
        
        equal_idxs = np.where(
            (data[original_tgt_label] == curr_target) &
            (data.index != row)
        )[0]
        negative_idxs = np.where(
            (data[original_tgt_label] != curr_target) &
            (data.index != row)
        )[0]

        sampled_equal_idxs = np.random.choice(equal_idxs, size=target_sample[curr_target], replace=False).tolist()
        sampled_negative_idxs = np.random.choice(negative_idxs, size=target_sample[1-curr_target], replace=False).tolist()
        sampled_idxs = sampled_equal_idxs + sampled_negative_idxs
        
        c_1_simulated += [row for _ in range(len(sampled_idxs))]
        c_2_simulated += sampled_idxs
    
    #take out duplicate
    initial_size = len(c_1_simulated)

    c_simulated = [[c_1_simulated[i], c_2_simulated[i]] for i in range(len(c_1_simulated))]
    c_simulated = [list(dist_x) for dist_x in set(tuple(x) for x in c_simulated)]
    if len(c_simulated) - initial_size > 0:
        print(f'reduces augmentation by: {len(c_simulated) - initial_size}')

    c_1_simulated = [c_1 for c_1, _ in c_simulated]
    c_2_simulated = [c_2 for _, c_2 in c_simulated]

    return c_1_simulated, c_2_simulated

def fe_new_col_name()->list:
    return [
        'mean_diff',
        'std_diff',
        'median_diff',
        'number_zero',
        'diff_mean',
        'diff_std',
        'diff_median',
    ]

def fe_pipeline(
        dataset_1: np.array, dataset_2: np.array,
        feature_list: list
    ) -> pd.DataFrame:

    dataset_contrast = pd.DataFrame(
        np.abs(
            dataset_1 - 
            dataset_2
        ), columns=feature_list
    )
    dataset_contrast['number_zero'] = (dataset_contrast == 0).mean(axis=0)
    dataset_contrast['mean_diff'] = dataset_contrast.mean(axis=0)
    dataset_contrast['std_diff'] = dataset_contrast.std(axis=0)
    dataset_contrast['median_diff'] = dataset_contrast.median(axis=0)

    dataset_contrast['diff_mean'] = np.abs(
        np.mean(dataset_1, axis=1) -
        np.mean(dataset_2, axis=1)
    )
    dataset_contrast['diff_std'] = np.abs(
        np.std(dataset_1, axis=1) -
        np.std(dataset_2, axis=1)
    )
    dataset_contrast['diff_median'] = np.abs(
        np.median(dataset_1, axis=1) -
        np.median(dataset_2, axis=1)
    )

    return dataset_contrast


def contrastive_pipeline(
        data: pd.DataFrame, feature_list: list, inference: bool,
        original_tgt_label: str, num_simulation: int = None
    ) -> pd.DataFrame:

    # c_1_simulated, c_2_simulated = get_all_combination(data, inference)
    c_1_simulated, c_2_simulated = get_stratified_example(data, original_tgt_label, inference)

    col_used = feature_list + [original_tgt_label]

    c_1_data = data.loc[
        c_1_simulated, col_used
    ].reset_index(drop=True)

    c_2_data = data.loc[
        c_2_simulated, col_used
    ].reset_index(drop=True)

    dataset_contrast = fe_pipeline(
        dataset_1=c_1_data, dataset_2=c_2_data,
        feature_list=feature_list
    )

    dataset_contrast['target_contrast'] = (
        c_2_data[original_tgt_label] == c_1_data[original_tgt_label]
    ).astype(int)

    print(f"Augmented {'Valid' if inference else 'Train'} dataset: {dataset_contrast.shape[0]} rows")
    return dataset_contrast