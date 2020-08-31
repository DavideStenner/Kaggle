# %% [code]
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
import math

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

def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = layers.Input(x_in.shape[1:])
        y = layers.Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return models.Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head

class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists

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
    def __init__(self, name, activation = 'swish', n_activation = 256):
        super(conv_block, self).__init__(name = name)
        
        self.conv = layers.Conv2D(round_filters(n_activation, 1, 1), 1, padding='same')
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

def extract_layer(
    backbone, final_embedding, num_classes,
    margin, logist_scale, dim_limit, filter_limit 
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
        
        if (dim_1<= dim_limit) & (n_filter_block < filter_limit):
            
            conv_block_dictionary[f'conv_block_{block}'] = conv_block(name = f'conv_block_{block}', n_activation = dimension_)

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

    normalized_embedding = tf.nn.l2_normalize(reduce_dimension, axis = 1)
    
    labels = layers.Input([], name='label')
    logist = ArcHead(num_classes = num_classes, margin = margin,
                     logist_scale = logist_scale)(normalized_embedding, labels)

    model_output = models.Model([input_layer, labels], logist)

    embedding_model = models.Model(input_layer, concatenated)
    
    return model_output, embedding_model

def SoftmaxLoss():
    """softmax loss"""
    def softmax_loss(y_true, y_pred):
        # y_true: sparse target
        # y_pred: logist
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)
        return tf.reduce_mean(ce)
    return softmax_loss

def build_model_extractor(optimizer, shape, num_classes,
                       model = 'efficientnet', modeltype = 'B0',
                       final_embedding = 2**10, loss = SoftmaxLoss(), dim_limit = 200, filter_limit = 1200,
                       weights = 'imagenet', trainable = False, margin=0.5, logist_scale=64):
    
    if model == 'efficientnet':
        backbone = load_effnet(shape, modeltype, weights = weights, trainable = trainable)
            
    model, embedding = extract_layer(backbone, final_embedding, num_classes, margin, logist_scale, dim_limit = dim_limit, filter_limit = filter_limit)
    
    model.compile(optimizer = optimizer, loss = loss)
        
    return model, embedding

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