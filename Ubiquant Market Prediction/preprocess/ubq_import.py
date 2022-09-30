import pandas as pd
import numpy as np
import os
import gc
import pickle
from ubq_utilities import reduce_mem_usage_sd

path_ubiquant = r'../input/ubiquant-market-prediction'

#read train
print('Starting reading train\n')

#start from float32
column_training = train = pd.read_csv(os.path.join(path_ubiquant, 'train.csv'), nrows = 1).columns
dtype = {x: 'O' if x == 'row_id' else 'float32' for x in column_training}

train = pd.read_csv(os.path.join(path_ubiquant, 'train.csv'), dtype = dtype)
gc.collect()

#downcast
print('Downcasting train\n')
train = reduce_mem_usage_sd(train)
gc.collect()

print('Saving dictionary map\n')
dtype_mapping = train.dtypes.to_dict()

train.to_parquet('train.parquet')

del train
gc.collect()

#import example
example_sample_submission = pd.read_csv(os.path.join(path_ubiquant, 'example_sample_submission.csv'))
example_test = pd.read_csv(os.path.join(path_ubiquant, 'example_test.csv'))

#downcast
print('Reading example dataset\n')
example_test = example_test.astype({x: y for x, y in dtype_mapping.items() if x in example_test.columns})

#save mapping file
print('Saving all\n')
with open('mapping_dict.pkl', 'wb') as file:
    pickle.dump(dtype_mapping, file)
    
#save dataset
example_sample_submission.to_parquet('example_sample_submission.parquet')
example_test.to_parquet('example_test.parquet')