import tensorflow as tf
import pandas as pd
import numpy as np
from ubq_utilities import STARTING_NUMERIC_FEAT_NAME
import tensorflow_probability as tfp

BATCH_SIZE = 4096

PARAMS = {
    'num_original_feature': len(STARTING_NUMERIC_FEAT_NAME),
    'hidden_units': [96, 96, 1024, 512, 256, 256, 64, 64], 
    'dropout_rates': [0.01, 0.05, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, .1], 
    'lr':1e-3, 
 }



def get_dataset(num_data, target, batch_size = BATCH_SIZE, train=True):
    def preprocess(X, y):
        return X, y
    dataset = tf.data.Dataset.from_tensor_slices((num_data, target)).map(preprocess)
    if train:
        dataset = dataset.shuffle(batch_size*4, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_ae_dataset(num_original_feat, all_feat, target, batch_size = BATCH_SIZE, train = True):
    original_feat = all_feat[:, :num_original_feat]
    
    original_dataset = tf.data.Dataset.from_tensor_slices(original_feat)
    feature_dataset = tf.data.Dataset.from_tensor_slices(all_feat)
    target_dataset = tf.data.Dataset.from_tensor_slices(target)

    output_dataset = tf.data.Dataset.zip((original_dataset, target_dataset, target_dataset))
    
    dataset = tf.data.Dataset.zip((feature_dataset, output_dataset))
    if train:
        dataset = dataset.shuffle(batch_size*4, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

class UbqSequenceMlp(tf.keras.utils.Sequence):

    def __init__(self, num_data, target, batch_size=4096, inference=False):
        self.x, self.y = num_data, target
        self.batch_size = batch_size
        self.nrows=self.x.shape[0]
        self.last_truncated = True if (self.nrows%self.batch_size) > 0 else false
        self.inference = inference

    def __len__(self):
        len_ = self.nrows // self.batch_size
        if self.last_truncated:
            return len_ + 1
        else:
            return len_
            
    def __getitem__(self, idx):
        inf_index = idx * self.batch_size
        sup_index = (idx + 1) * self.batch_size
        
        if idx < self.__len__():
            batch_x = self.x[
                inf_index:sup_index, :
            ]
            batch_y = self.y[
                inf_index:sup_index
            ]
        else:
            batch_x = self.x[
                inf_index:, :
            ]
            batch_y = self.y[
                inf_index:
            ]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.inference:
            pass
        else:
            idx = np.array(range(self.x.shape[0]))
            np.random.shuffle(idx)

            self.x, self.y = self.x[idx, :], self.y[idx]

            assert self.x.shape[0] == self.nrows

class UbqSequence(tf.keras.utils.Sequence):

    def __init__(self, num_data, target, batch_size=4096, inference=False, num_original_feat=len(STARTING_NUMERIC_FEAT_NAME)):
        self.x, self.y = num_data, target
        self.batch_size = batch_size
        self.nrows=self.x.shape[0]
        self.last_truncated = True if (self.nrows%self.batch_size) > 0 else false
        self.inference = inference
        self.num_original_feat=num_original_feat

    def __len__(self):
        len_ = self.nrows // self.batch_size
        if self.last_truncated:
            return len_ + 1
        else:
            return len_
            
    def __getitem__(self, idx):
        inf_index = idx * self.batch_size
        sup_index = (idx + 1) * self.batch_size
        
        if idx < self.__len__():
            batch_x = self.x[
                inf_index:sup_index, :
            ]
            batch_y = self.y[
                inf_index:sup_index
            ]
        else:
            batch_x = self.x[
                inf_index:, :
            ]
            batch_y = self.y[
                inf_index:
            ]

        return batch_x, [batch_x[:, :self.num_original_feat], batch_y, batch_y]

    def on_epoch_end(self):
        if self.inference:
            pass
        else:
            idx = np.array(range(self.x.shape[0]))
            np.random.shuffle(idx)

            self.x, self.y = self.x[idx, :], self.y[idx]

            assert self.x.shape[0] == self.nrows

def tfp_correlation(x, y, sample_axis=0):
    corr = tfp.stats.correlation(x, y, sample_axis=sample_axis, event_axis=None)
    return corr

def create_mlp(
    shape, steps,
    units=[1024, 512, 256, 256, 64, 64],
    dropouts=[.35, .35, .35, .35, .1, .1],
    metrics={'output': tfp_correlation}, lr=0.0008,
    compile_other_params = {}
):
    
    def fc_block(x, unit, dropout):
        x = tf.keras.layers.Dense(unit)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    inp = tf.keras.layers.Input((shape))
    x = tf.keras.layers.GaussianNoise(0.01)(inp)
    
    for layer in range(len(units)):
        x = fc_block(x, unit = units[layer], dropout=dropouts[layer])
                                                  
    output = tf.keras.layers.Dense(1, activation = 'linear', name='output')(x)
    model = tf.keras.models.Model(inputs = [inp], outputs = [output])
    scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate = lr, decay_steps = steps, end_learning_rate = 0.000005
    )
    opt = tf.keras.optimizers.Adam(learning_rate = scheduler)
    model.compile(
        optimizer = opt,
        loss = [tf.keras.losses.MeanSquaredError()],
        metrics = metrics,
        **compile_other_params
    )
    return model

def create_ae_mlp(
    num_original_feature, num_total_feature, hidden_units, dropout_rates, 
    steps,
    lr=0.0008,
    metrics = {
            'decoder': tf.keras.metrics.MeanSquaredError(name = 'MSE'), 
            'ae_output': tf.keras.metrics.MeanSquaredError(name = 'MSE'), 
            'output': tf.keras.metrics.MeanSquaredError(name = 'MSE'), 
        },
    compile_other_params = {}
    ):
    
    def fc_block(x, unit, dropout):
        x = tf.keras.layers.Dense(unit)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x
    
    inp = tf.keras.layers.Input(shape = (num_total_feature, ), name='num_input')
    
    x0 = tf.keras.layers.BatchNormalization()(inp)
    
    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)
    
    decoder = fc_block(encoder, num_original_feature, dropout_rates[1])
    x_decoder = tf.keras.layers.Dense(num_original_feature, activation = 'linear', name = 'decoder')(decoder)

    x_ae = fc_block(decoder, hidden_units[1], dropout_rates[2])

    out_ae = tf.keras.layers.Dense(1, activation = 'linear', name = 'ae_output')(x_ae)
    
    x = tf.keras.layers.Concatenate()([x0, encoder])
    
    for i in range(2, len(hidden_units)):
        x = fc_block(x, hidden_units[i], dropout_rates[i + 1])
        
    out = tf.keras.layers.Dense(1, activation = 'linear', name = 'output')(x)
    scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate = lr, decay_steps = steps, end_learning_rate = 0.000005
    )
    opt = tf.keras.optimizers.Adam(learning_rate = scheduler)

    model = tf.keras.models.Model(inputs = [inp], outputs = [x_decoder, out_ae, out])
    model.compile(
        optimizer = opt,
        loss = {
            'decoder': tf.keras.losses.MeanSquaredError(), 
            'ae_output': tf.keras.losses.MeanSquaredError(),
            'output': tf.keras.losses.MeanSquaredError(), 
        },
        metrics = metrics, 
        **compile_other_params
    )
    
    return model

def create_conv(
    shape, steps,
    units=[1024, 512, 256, 256, 64, 64],
    dropouts=[.35, .35, .35, .35, .1, .1],
    filters=[16, 32, 64, 128, 256],
    dropout_conv = [.35, .35, .35, .35, .35],
    stride=[1, 4, 4, 4, 4],
    metrics={'output': tfp_correlation}, lr=0.0008,
    compile_other_params = {}
):
    
    def fc_block(x, unit, dropout):
        x = tf.keras.layers.Dense(unit)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x
    
    def conv_block(x, filters, dropout, stride, kernel_size=4):
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x
    
    inp = tf.keras.layers.Input((shape))
    x = tf.keras.layers.GaussianNoise(0.01)(inp)
    
    x = fc_block(x, unit=units[0], dropout=dropouts[0])
    x = tf.keras.layers.Reshape((-1,1))(x)
    
    for conv in range(len(dropout_conv)):
        x = conv_block(x, filters[conv], dropout_conv[conv], stride[conv])
    
    x = tf.keras.layers.Flatten()(x)

    for layer in range(len(units)):
        x = fc_block(x, unit = units[layer], dropout=dropouts[layer])
                                                  
    output = tf.keras.layers.Dense(1, activation = 'linear', name='output')(x)
    model = tf.keras.models.Model(inputs = [inp], outputs = [output])
    scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate = lr, decay_steps = steps, end_learning_rate = 0.000005
    )
    opt = tf.keras.optimizers.Adam(learning_rate = scheduler)
    model.compile(
        optimizer = opt,
        loss = [tf.keras.losses.MeanSquaredError()],
        metrics = metrics,
        **compile_other_params
    )
    return model

class SharpeCorrScore(tf.keras.callbacks.Callback):
    
    def __init__(self, model, time_id, test_input, test_output, step = 1):
        
        self.model = model        
        self.step = step
        self.time_id = time_id
        self.test_input = test_input
        self.test_output = test_output
        
    def calculate_corr(
        self,
        df: pd.DataFrame, 
        time_col: str='time_id', pred_col: str='y_pred', 
        target_col: str='y_true'
    ):
        
        df_used = df[[time_col, pred_col, target_col]]
        corr_by_time_id = df.groupby(time_col).apply(
            lambda x: x[[pred_col]].corrwith(x[target_col])
        )

        mean_corr = corr_by_time_id.mean().values[0]
        std_corr = corr_by_time_id.std().values[0]
        
        sharpe_corr = mean_corr/std_corr
                
        return mean_corr, sharpe_corr

    def corr_sharp(
        self,
        y_pred: np.array, y_true: np.array, 
    ) -> float:
        """
        Pearson correlation coefficient metric
        """
        pd_info = pd.DataFrame(
            {
                'time_id': self.time_id,
                'y_pred': y_pred,
                'y_true': y_true
            }
        )
        mean_corr, sharpe_corr = self.calculate_corr(pd_info)
        
        return mean_corr, sharpe_corr

    def on_epoch_end(self, epoch, logs):
        
        if (self.step == 1) or ((epoch > 0) & (epoch % self.step == 0)):

            pred = self.model.predict(self.test_input)
            y_pred = np.reshape(pred[-1], (-1))
            
            mean_corr, sharpe_corr = self.corr_sharp(y_pred, self.test_output)

            logs['mean_corr'] = mean_corr
            logs['sharpe_corr'] = sharpe_corr

            print("Step: {} -- sharpe_corr: {} -- mean_corr: {}".format(epoch, logs['sharpe_corr'], logs['mean_corr']))