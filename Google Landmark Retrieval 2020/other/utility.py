import os
os.system('pip install -q efficientnet --quiet')

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import itertools
from tensorflow.keras.applications.imagenet_utils import preprocess_input

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, batch_size, image_size, cropper, preprocess):
        
        #batch size and image size
        self.batch_size = batch_size
        self.image_size = image_size

        #list of landmark id in train
        self.landmark_ids = dataset["landmark_id"].unique()
        
        #list of id of every image in train
        self.ids = dataset["id"].unique()
        
        #dictionary to pass from landmark to images
        self.dict_landmark_to_images_mapping = dataset.groupby("landmark_id")["id"].apply(list).to_dict()
        
        #path of each image 
        self.path_dict = dataset.set_index("id")["train_path"].to_dict()
        
        #shuffle index
        self.on_epoch_begin()
        
        self.cropper = cropper
        self.preprocess = preprocess
        
        pass
    
    #number of landmark divided batch_size
    def __len__(self,):
        return int(np.floor(len(self.landmark_ids) / self.batch_size))

    #at begin of train shuffle landmark id
    def on_epoch_begin(self):
        self.indexes = self.landmark_ids.copy()
        np.random.shuffle(self.indexes)
        
    def __getitem__(self, index):

        #get landmark id after shuffle
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #get anchors from landmark id
        anchors = [self.get_anchor(x) for x in indexes]
        
        #get positive from anchor
        positives = [self.get_positives(anchors[x], indexes[x]) for x in range(len(indexes))]
        
        #get negative from anchor
        negatives = [self.get_negatives(anchors[x], indexes[x]) for x in range(len(indexes))]
        
        #define list of list of path to images
        X = [anchors,positives,negatives]
        
        #get images
        X = self.get_images(X)
        
        y = indexes
        return X, y
    
    #Get anchor from landmark
    def get_anchor(self, landmark_id):
        
        #all image of landmark
        all_landmark_image = self.dict_landmark_to_images_mapping[landmark_id]
        
        #select one image
        return np.random.choice(all_landmark_image)
    
    #get positive from anchor inside landmark_id group
    def get_positives(self, anchor, landmark_id):
        
        #get all image relative to landmark id group
        all_positive_images = self.dict_landmark_to_images_mapping[landmark_id]
        
        #find positive image not considering anchor one.
        all_positive_images = [t for t in all_positive_images if t != anchor]
        
        #return random choice from positive images
        return np.random.choice(all_positive_images)
    
    #get negative from anchor outside landmark_id group
    def get_negatives(self, anchor, landmark_id):
        
        #choose random landmark_id (not the anchor one)
        random_negative_landmarkid = np.random.choice([t for t in self.landmark_ids if t != landmark_id]) ##CAN INSERT LOGIC INSIDE RANDOM !
        
        #choose random image inside this group
        random_negative_image = self.dict_landmark_to_images_mapping[random_negative_landmarkid]
        
        return np.random.choice(random_negative_image)
    
    #pre process
    def eff_net_preprocess(self, image):
        #center and crop
        image = self.cropper(image, image_size = self.image_size)
        
        #pre process
        image = self.preprocess(image)
        return image
    
    #function used to get images from list of id of images
    def get_images(self, X):
        X_=[]
        
        anchors = X[0]
        positives = X[1]
        negatives = X[2]
        
        anchors_ = []
        positives_ = []
        negatives_ = []
        
        #get anchor images :  Convert ids --> Path of image --> Download locally image --> Pre process --> Append
        for a in anchors:
            
            a = self.path_dict[a]
            a = cv2.imread(a)
            a = self.eff_net_preprocess(a)
            anchors_.append(a)
        
        for p in positives:
            
            p = self.path_dict[p]
            p = cv2.imread(p)
            p = self.eff_net_preprocess(p)
            positives_.append(p)
        
        for n in negatives:
            
            n = self.path_dict[n]
            n = cv2.imread(n)
            n = self.eff_net_preprocess(n)
            negatives_.append(n)
        
        X_ = [np.array(anchors_), np.array(positives_), np.array(negatives_)]
        return X_
    
class DataGenerator_mining(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, batch_size, image_size, number_of_image = 10):
        
        self.number_of_image = number_of_image
        
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
        
        #shuffle index
        self.on_epoch_begin()
                
        pass
    
    #number of landmark divided batch_size
    def __len__(self,):
        return int(np.floor(len(self.landmark_ids) / self.batch_size))

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
        image = tf.image.resize(image, (self.image_size, self.image_size))
        
        #pre process
        image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode = 'torch')
        
        image = image.numpy()
        
        return image
    
    #function used to get images from list of id of images
    def get_images(self, X):
        
        X_=[]
        y_ = []
        for id_, target in X:
            
            path = self.path_dict[id_]
            
            images = cv2.imread(path)
            images = self.eff_net_preprocess(images)
            
            X_.append(images)
            y_.append(target)
        X_, y_ = np.array(X_), np.array(y_)
        return X_, y_

class DataGenerator_mining_cluster(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, batch_size, image_size, cropper, preprocess, number_of_image = 10):
        
        self.number_of_image = number_of_image
        
        #batch size and image size
        self.batch_size = batch_size // self.number_of_image
            
        self.image_size = image_size

        #list of landmark id in train
        self.landmark_group = dataset[["landmark_id", "cluster_id"]].drop_duplicates().reset_index(drop = True)
        self.indexes = self.landmark_group.landmark_id
        
        #list of id of every image in train
        self.ids = dataset["id"].unique()
        
        #dictionary to pass from landmark to images
        self.dict_landmark_to_images_mapping = dataset.groupby("landmark_id")["id"].apply(list).to_dict()
        
        #path of each image 
        self.path_dict = dataset.set_index("id")["train_path"].to_dict()
                
        self.cropper = cropper
        self.preprocess = preprocess
        
        pass
    
    #number of landmark divided batch_size
    def __len__(self,):
        return int(np.floor(len(self.indexes) / self.batch_size))

    #at begin of train shuffle landmark id
    def on_epoch_begin(self):
        self.landmark_group = self.landmark_group.groupby('cluster_id').apply(lambda x: x.sample(frac = 1, replace = False)).reset_index(drop = True)
        self.indexes = self.landmark_group.landmark_id
        
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
        image = self.cropper(image, image_size = self.image_size)
        
        #pre process
        image = self.preprocess(image)
        return image
    
    #function used to get images from list of id of images
    def get_images(self, X):
        
        X_=[]
        y_ = []
        for id_, target in X:
            
            path = self.path_dict[id_]
            
            images = cv2.imread(path)
            images = self.eff_net_preprocess(images)
            
            X_.append(images)
            y_.append(target)
            
        X_, y_ = np.array(X_), np.array(y_)
        return X_, y_

class predictDataset(tf.keras.utils.Sequence):
    
    def __init__(self, path, batch_size, image_size):
        self.paths = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.indexes = np.arange(len(self.paths))
        self.len_index = len(self.indexes)

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
        images=[cv2.imread(t) for t in images]
        images=[self.eff_net_preprocess(t) for t in images]
        return np.array(images)

    def eff_net_preprocess(self, image):
        #center and crop
        image = tf.image.resize(image, (self.image_size, self.image_size))
        
        #pre process
        image = tf.keras.applications.imagenet_utils.preprocess_input(image)
        
        image = image.numpy()
        
        return image
