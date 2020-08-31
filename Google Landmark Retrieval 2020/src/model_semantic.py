import os

os.system('pip install -q efficientnet --quiet')
os.system('pip install -q image-classifiers --quiet')

from classification_models.tfkeras import Classifiers
import tensorflow as tf
from tensorflow.keras import layers,models
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB3, EfficientNetB5
from sklearn.metrics.pairwise import euclidean_distances
import time
from transformers import WarmUp
import re
import numpy as np

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

MODEL_INPUT = {
    'B0': 224,
    'B1': 240,
    'B2': 260,
    'B3': 300,
    'B4': 380,
    'B5': 456,
    'B6': 528,
    'B7': 600,
}

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


class GEM(tf.keras.layers.Layer):
    def __init__(self, name, init_p = 1):
        super(GEM, self).__init__(name = name)
        assert init_p >0.1
        self.eps = 1e-6
        self.init_p = init_p
        
    def build(self, input_shape):
        self.p = self.add_weight(
            name='power',
            shape=[1],
            initializer = tf.keras.initializers.RandomUniform(minval = self.init_p - 0.1, maxval = self.init_p + 0.1),
            trainable = True,
            constraint = tf.keras.constraints.NonNeg()
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.math.pow(tf.math.maximum(self.eps, inputs), self.p)
        size = (x.shape[1], x.shape[2])
        
        x = tf.math.pow(tf.nn.avg_pool2d(x, size, 1, padding='VALID'), 1./self.p)
        
        return x

class ChannelSE(tf.keras.Model):
    def __init__(self, gem, channels, name, reduction=16, activation = 'sigmoid', channels_axis = 3):
        super(ChannelSE, self).__init__(name = name)
        
        self.reduction = reduction
        self.activation = activation
        self.channels_axis = channels_axis
        
        self.conv1 = layers.Conv2D(channels // self.reduction, (1, 1), kernel_initializer='he_uniform')
        self.conv2 = layers.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')
        self.gem = gem
        
    def expand_dims(self, x, channels_axis):
        return x[:, None, None, :]
    
    def call(self, inputs):
        x = inputs

        # squeeze and excitation block in PyTorch style with
        x = self.gem(x)
        x = layers.Flatten()(x)
        x = layers.Lambda(self.expand_dims, arguments={'channels_axis': self.channels_axis})(x)
        x = self.conv1(x)
        x = layers.Activation('swish')(x)
        x = self.conv2(x)
        x = layers.Activation(self.activation)(x)

        # apply attention
        x = layers.Multiply()([inputs, x])
        return x

class conv_block(tf.keras.Model):
    def __init__(self, name, activation = 'swish', n_activation = 256, kernel_size = 1):
        super(conv_block, self).__init__(name = name)
        
        self.conv = layers.Conv2D(round_filters(n_activation, 1, 1), kernel_size, padding='same')
        self.activation = activation
    
        self.batch_norm = layers.BatchNormalization(axis = 3)
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = layers.Activation(self.activation)(x)
        return x

class flatten_block(tf.keras.Model):
    def __init__(self, name, reduced_dimension = 256, reduce = True):
        super(flatten_block, self).__init__(name = name)
        self.reduce = reduce
        
        if reduce:
            self.dense = layers.Dense(reduced_dimension)
        
    def call(self, inputs):
        x = layers.Flatten()(inputs)
        x = tf.nn.l2_normalize(x, axis = 1)
        if self.reduce:
            x = layers.Dense(reduced_dimension)(x)
        return x

def load_effnet(shape, efftype = 'B0', weights = 'imagenet', trainable = False):
    
    if efftype == 'B0':
        backbone = EfficientNetB0(weights = weights, include_top = False, input_shape = shape)
        
    if efftype == 'B3':
        backbone = EfficientNetB3(weights = weights, include_top = False, input_shape = shape)

    if efftype == 'B5':
        backbone = EfficientNetB5(weights = weights, include_top = False, input_shape = shape)

    for layer in backbone.layers:
        layer.trainable = trainable
    
    return backbone

def load_resnet(shape, efftype = 'R50', weights = 'imagenet'):
    typeof = efftype[1:]
    loader, _ = Classifiers.get(f'resnet{typeof}')
    backbone = loader(shape, weights = weights, include_top = False)

    input_layer = backbone.input

    for layer in backbone.layers:
        layer.trainable = False
    
    return backbone

def create_Model(
    backbone, dimension = 1024, channel_axis = 3, reduce_factor = 2,
    num_block = 4
    ):
        
    assert num_block % 2 == 0

    n_activation = int(dimension/reduce_factor)

    conv_block_dictionary = {}
    squeeze_block_dictionary = {}
    gem_block_dictionary = {}
    flatten_block_dictionary = {}

    prob_activation = ['sigmoid' for x in range(num_block//2)] + ['softmax' for x in range(num_block//2)]
    init_p = [1 - 1/(2 + x) for x in range(num_block//2)] + [2 + x for x in range(num_block//2)]

    for block in range(num_block):
            conv_block_dictionary[f'conv_block_{block}'] = conv_block(name = f'conv_block_{block}', n_activation = n_activation)

            gem_block_dictionary[f'gem_block_{block}'] = GEM(name = f'gem_block_{block}', init_p = init_p[block])

            squeeze_block_dictionary[f'squeeze_block_{block}'] = ChannelSE(
                gem = gem_block_dictionary[f'gem_block_{block}'],
                channels = n_activation, name = f'squeeze_block_{block}',
                activation = prob_activation[block]
            )
            flatten_block_dictionary[f'flatten_block_{block}'] = flatten_block(name = f'flatten_block_{block}', reduce = False)
    
    input_layer = backbone.input

    output_effnet = backbone.output
    
    list_head = []
    for block in range(num_block):

        x = conv_block_dictionary[f'conv_block_{block}'](output_effnet)

        x = squeeze_block_dictionary[f'squeeze_block_{block}'](x)

        x = gem_block_dictionary[f'gem_block_{block}'](x)

        x = flatten_block_dictionary[f'flatten_block_{block}'](x)

        list_head += [x]

    concatenated = layers.concatenate(list_head)

    reduce_dimension = layers.Dense(dimension)(concatenated)

    output_normalized = tf.nn.l2_normalize(reduce_dimension, axis = 1)
    
    model_output = models.Model(inputs = input_layer, outputs = output_normalized)
    return model_output

def build_model_extractor(optimizer, shape, loss,
                       model = 'efficientnet', modeltype = 'B0',
                       final_embedding = 2**10, kernel_size = 1,
                       weights = 'imagenet', trainable = False):
               
    if model == 'efficientnet':
        backbone = load_effnet(shape, modeltype, weights = weights, trainable = trainable)
            
    model = extract_layer(backbone, final_embedding, kernel_size = kernel_size)
    
    model.compile(optimizer = optimizer, loss = loss)
        
    return model

def extract_layer(
    backbone, final_embedding, kernel_size
    ):

    step_layer_name = []

    step_block = np.unique([int(re.search("\d", x.name).group()) for x in backbone.layers[5:-3]])

    for layer in step_block:
        step_layer_name.append([x.name for x in backbone.layers if f'block{layer}' in x.name][-1])
    
    conv_block_dictionary = {}
    squeeze_block_dictionary = {}
    gem_block_dictionary = {}
    flatten_block_dictionary = {}

    num_block = len(step_layer_name)

    prob_activation = ['softplus'] * num_block
    init_p = [3] * num_block
    
    block_used = 0
    for block in range(num_block):
            
        n_filter_block = backbone.get_layer(step_layer_name[block]).output.shape[-1]
        dim_1 = backbone.get_layer(step_layer_name[block]).output.shape[1]
        
        if (dim_1<=100) & (n_filter_block < 1000):
            block_used += 1
            
    dimension_ = (final_embedding * 1.5)//block_used
    
    for block in range(num_block):
            
        n_filter_block = backbone.get_layer(step_layer_name[block]).output.shape[-1]

        dim_1 = backbone.get_layer(step_layer_name[block]).output.shape[1]
        
        if (dim_1<=100) & (n_filter_block < 1000):
            
            conv_block_dictionary[f'conv_block_{block}'] = conv_block(name = f'conv_block_{block}', n_activation = dimension_, kernel_size = kernel_size)

            gem_block_dictionary[f'gem_block_{block}'] = GEM(name = f'gem_block_{block}', init_p = init_p[block])

            squeeze_block_dictionary[f'squeeze_block_{block}'] = ChannelSE(
                gem = gem_block_dictionary[f'gem_block_{block}'],
                channels = dimension_, name = f'squeeze_block_{block}',
                activation = prob_activation[block]
            )
            flatten_block_dictionary[f'flatten_block_{block}'] = flatten_block(name = f'flatten_block_{block}', reduce = False)

    input_layer = backbone.input

    list_head = []
    for block in range(num_block):
        
        output_block = backbone.get_layer(step_layer_name[block]).output
        dim_1 = backbone.get_layer(step_layer_name[block]).output.shape[1]
        
        if (dim_1<=100) & (n_filter_block < 1000):

            x = conv_block_dictionary[f'conv_block_{block}'](output_block)

            x = squeeze_block_dictionary[f'squeeze_block_{block}'](x)

            x = gem_block_dictionary[f'gem_block_{block}'](x)

            x = flatten_block_dictionary[f'flatten_block_{block}'](x)

            list_head += [x]
    
    concatenated = layers.concatenate(list_head)

    reduce_dimension = layers.Dense(final_embedding)(concatenated)

    output_normalized = tf.nn.l2_normalize(reduce_dimension, axis = 1)

    model_output = models.Model(inputs = input_layer, outputs = output_normalized)

    return model_output

def build_model_mining(optimizer, shape, loss,
                       model = 'efficientnet', modeltype = 'B0',
                       dimension = 1024, reduce_factor = 2, num_block = 4,
                       weights = 'imagenet', trainable = False):
               
    if model == 'efficientnet':
        backbone = load_effnet(shape, modeltype, weights = weights, trainable = trainable)
        
    elif model == 'resnet':
        backbone = load_resnet(shape, modeltype, weights = weights, trainable = trainable)
    
    model = create_Model(backbone, dimension = dimension, reduce_factor = reduce_factor, num_block = num_block)
        
    model.compile(optimizer = optimizer, loss = loss)
        
    return model

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.time()
        
    def on_epoch_end(self,epoch,logs = {}):
        logs['time_epoch'] = time.time() - self.timetaken
        self.timetaken = time.time()
        
class Score_call(tf.keras.callbacks.Callback):
    
    def __init__(self, embedding_model, index_dataset, query_dataset, index_id, query_id, real_dic, step = 5):
        
        self.embedding_model = embedding_model
                
        self.index_dataset = index_dataset
        self.query_dataset = query_dataset
                        
        self.index_id = index_id
        self.query_id = query_id
        
        self.real_dic = real_dic
        self.MAP_early_stop = 0
        
        self.step = step
        
    def MeanAveragePrecision(self, predictions, retrieval_solution, max_predictions):
        """Computes mean average precision for retrieval prediction.
        Args:
        predictions: Dict mapping test image ID to a list of strings corresponding
          to index image IDs.
        retrieval_solution: Dict mapping test image ID to list of ground-truth image
          IDs.
        max_predictions: Maximum number of predictions per query to take into
          account. For the Google Landmark Retrieval challenge, this should be set
          to 100.
        Returns:
        mean_ap: Mean average precision score (float).
        Raises:
        ValueError: If a test image in `predictions` is not included in
          `retrieval_solutions`.
        """
        # Compute number of test images.
        num_test_images = len(retrieval_solution.keys())

        # Loop over predictions for each query and compute mAP.
        mean_ap = 0.0
        for key, prediction in predictions.items():
            if key not in retrieval_solution:
                raise ValueError('Test image %s is not part of retrieval_solution' % key)

            # Loop over predicted images, keeping track of those which were already
            # used (duplicates are skipped).
            ap = 0.0
            already_predicted = set()
            num_expected_retrieved = min(len(retrieval_solution[key]), max_predictions)
            num_correct = 0
            for i in range(min(len(prediction), max_predictions)):
                if prediction[i] not in already_predicted:
                    if prediction[i] in retrieval_solution[key]:
                        num_correct += 1
                        ap += num_correct / (i + 1)
                    already_predicted.add(prediction[i])

            ap /= num_expected_retrieved
            mean_ap += ap

        mean_ap /= num_test_images

        return mean_ap

    def knn_lookup(self, embedding_index, embedding_query, max_predictions):
        distance = euclidean_distances(embedding_query, embedding_index)
        
        top_k = distance.argsort(axis = 1)[:, :max_predictions]
        top_k_index_id = self.index_id[top_k]
        
        prediction_dic = {}
        for n, id_ in enumerate(self.query_id):
            prediction_dic[id_] = top_k_index_id[n, :].tolist()
        
        return prediction_dic
            

    def on_epoch_end(self, epoch, logs):
        
        if (self.step == 1) or ((epoch > 0) & (epoch % self.step == 0)):
            score_time = time.time()

            embedding_index = self.embedding_model.predict(self.index_dataset)
            embedding_query = self.embedding_model.predict(self.query_dataset)

            MAP_list = []

            time_step = time.time()

            for _, k in enumerate([50, 100, 200, 300]):
                prediction_dic = self.knn_lookup(embedding_index, embedding_query, k)
                MAP_k = self.MeanAveragePrecision(prediction_dic, self.real_dic, k) 
                logs[f'validation_MAP{k}'] = MAP_k
                MAP_list += [MAP_k]

                if _ == 0:
                    time_step = time.time() - time_step


            logs['time_step_lookup'] = time_step

            score_time = time.time() - score_time
            logs['score_time'] = score_time


            print("Step: {} -- Loss: {}\nMAP_50: {} -- MAP_100: {} -- MAP_200: {} -- MAP_300: {}\n".format(epoch, logs['loss'], *MAP_list))
            print('Time Training Epoch: {} -- Time Look Up: {} -- Time Single Map Est.: {}'.format(logs['time_epoch'], score_time, time_step))
            print('-' * 180, '\n')

            #CHECK THAT MODEL DOESN'T COLLAPSE OTHERWISE STOP TRAIN
            if (MAP_list[1] <= 0.2) or (MAP_list[1] < self.MAP_early_stop):

                self.embedding_model.stop_training = True
                print('Stopping training due to collapse...\n')

            else:

                self.MAP_early_stop = MAP_list[1]
        else:
            print("Step: {} -- Loss: {}\n".format(epoch, logs['loss']))
            print('Time Training Epoch: {} -- Time Look Up: {} -- Time Single Map Est.: {}'.format(logs['time_epoch']))
            print('-' * 180, '\n')
            
def linear_warmup(init_lr, num_train_steps, num_warmup_steps, min_lr_ratio = 0):
    
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate = init_lr,
        decay_steps = num_train_steps - num_warmup_steps,
        end_learning_rate = init_lr * min_lr_ratio,
    )
    lr_schedule = WarmUp(
        initial_learning_rate = init_lr, decay_schedule_fn = lr_schedule, warmup_steps = num_warmup_steps,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule, epsilon = 1e-8)
    
    return optimizer