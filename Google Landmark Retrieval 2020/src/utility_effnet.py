import os
os.system('pip install -q efficientnet --quiet')

import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
from tensorflow.keras.applications.imagenet_utils import preprocess_input as _preprocess_input
import cv2

CROP_PADDING = 32

def preprocess_effnet(image_tensor,
                        image_size):
    """Preprocesses the given image for evaluation.
    Args:
    image_tensor: `Tensor` representing an image of arbitrary size.
    image_size: image size.
    Returns:
    A preprocessed image `Tensor`.
    """
    image = _decode_and_center_crop(image_tensor, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype = tf.float32)

    return image

def _decode_and_center_crop(image_tensor, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = image_tensor.shape
    
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    
    image = tf.image.crop_to_bounding_box(image_tensor, offset_height, offset_width, padded_center_crop_size, padded_center_crop_size)

    image = _resize_image(image, image_size)
    return image

def _resize_image(image, image_size):
    return tf.image.resize([image], [image_size, image_size], method = tf.image.ResizeMethod.BICUBIC)[0]

class DataGenerator_mining(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, batch_size, image_size, number_of_image = 10, preprocess_effnet = preprocess_effnet, num_step = None):
        
        self.number_of_image = number_of_image
        
        self.preprocess_effnet = preprocess_effnet
        
        #batch size and image size
        self.batch_size = batch_size // self.number_of_image
            
        self.image_size = image_size

        #list of landmark id in train
        self.landmark_ids = dataset["landmark_id"].unique()
        
        #list of id of every image in train
        self.ids = dataset["id"].unique()
        
        #dictionary to pass from landmark to images
        self.dict_landmark_to_images_mapping = dataset.groupby("landmark_id")["id"].apply(list).to_dict()
        
        #path of each image 
        self.path_dict = dataset.set_index("id")["train_path"].to_dict()
        
        self.num_step = num_step
        
        #shuffle index
        self.on_epoch_begin()
         
        pass
    
    #number of landmark divided batch_size
    def __len__(self,):
        if self.num_step is None:
            return int(np.floor(len(self.landmark_ids) / self.batch_size))
        else:
            return self.num_step
        
    #at begin of train shuffle landmark id
    def on_epoch_begin(self):
        self.indexes = self.landmark_ids.copy()
        np.random.shuffle(self.indexes)
        
    def __getitem__(self, index):

        #get landmark id after shuffle
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #get image id from lanmark_id
        images_info = [self.get_id(x) for x in indexes]
        images_info_flatted = list(itertools.chain.from_iterable(images_info))
                
        #get images
        X, y = self.get_images(images_info_flatted)
        
        return X, y
    
    #Get anchor from landmark
    def get_id(self, landmark_id):
        
        #all image of landmark
        all_landmark_image = self.dict_landmark_to_images_mapping[landmark_id]
        
        #select k images
        images_id = np.random.choice(all_landmark_image, self.number_of_image, replace = False)
        images_landmark = np.ones(self.number_of_image) * landmark_id
        
        zipped_ = list(zip(list(images_id), list(images_landmark)))
        
        return zipped_
        
    #pre process
    def eff_net_preprocess(self, image):
        
        #center and crop
        image = self.preprocess_effnet(image, self.image_size)
        
        #pre process
        image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode = 'torch')
                
        return image
    
    #function used to get images from list of id of images
    def get_images(self, X):
        
        X_ = []
        y_ = []
        for id_, target in X:
            
            path = self.path_dict[id_]
            
            images = self.reader(path)
            images = self.eff_net_preprocess(images)
            
            X_.append(images)
            y_.append(target)
            
        X_, y_ = tf.stack(X_), tf.stack(y_)
        return X_, y_
    
    def reader(self, path):
        image_bytes = tf.io.read_file(path)
        images = tf.io.decode_jpeg(image_bytes, channels = 3)
        return(images)

class predictDataset(tf.keras.utils.Sequence):
    
    def __init__(self, path, batch_size, image_size, preprocess_effnet = preprocess_effnet):
        
        self.paths = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.indexes = np.arange(len(self.paths))
        self.len_index = len(self.indexes)

        self.preprocess_effnet = preprocess_effnet
        
    def __len__(self,):
        
        if np.mod(self.len_index, self.batch_size) == 0:
            
            self.last_index = int(self.len_index/self.batch_size) - 1
            self.truncate = False
            
            return self.last_index + 1
        
        else:
            
            self.last_index = int(np.floor(len(self.paths) / self.batch_size))
            self.truncate = True
            
            return self.last_index + 1
        
    def __getitem__(self, index):
        if (index == self.last_index) & self.truncate:
            indexes = self.indexes[index*self.batch_size:]

        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            
        images=[self.paths[t] for t in indexes]
        images=[self.reader(t) for t in images]
        images=[self.eff_net_preprocess(t) for t in images]
        return tf.stack(images)

    #pre process
    def eff_net_preprocess(self, image):
        
        #center and crop
        image = self.preprocess_effnet(image, self.image_size)
        
        #pre process
        image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode = 'torch')
                
        return image

    def reader(self, path):
        images = cv2.imread(path) #tf.io.read_file
        return(images)
