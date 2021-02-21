import os, sys

import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm, tqdm_notebook
from glob import glob
import time
import itertools
import nvgpu
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import cv2
import imgaug.augmenters as iaa

import matplotlib.font_manager as fm
fontprop = fm.FontProperties(fname='/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf', size=18)

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import confusion_matrix

# Model
import tensorflow as tf
import keras
from keras.preprocessing import image

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras.models import model_from_json 

# 이용가능 한 GPU 확인
def ckech_available_gpu(num_gpu=1, min_mem_mb=8000):
    gpu_info=nvgpu.gpu_info()
    key_list = gpu_info[0].keys()

    gpu_info_dict={}
    for key in key_list :
        gpu_info_dict[key] = []

    for elem in gpu_info :
        for key in key_list :
            gpu_info_dict[key].append(elem[key])

    gpu_info_df = pd.DataFrame(gpu_info_dict)
    gpu_info_df["mem_free"] = gpu_info_df["mem_total"]*(1-gpu_info_df["mem_used_percent"]/100)

    available_list=gpu_info_df.loc[gpu_info_df["mem_free"]>min_mem_mb].sort_values("mem_used")["index"].values
            
    return available_list[:num_gpu]

class Config(object):
    GPU_OPTION="auto"
    GPU_NUM=1
    GPU_MIN_MEM=8000 # mb
    ALLOW_CPU=False

    MODEL_NAME="InceptionResNetV2"
    CLASS_WEIGHTS=True
    BATCH_SIZE=32
    NUM_EPOCH=200
    LR=None
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
    def to_dict(self):
        config_dict={}
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                config_dict[a]=getattr(self, a)
                
        return config_dict
        
def build_model(model_config, base_model):
    _base_model=base_model(include_top=False, 
                           weights='imagenet',
                           pooling='avg',
                           input_shape=model_config.INPUT_SHAPE)
    _base_model.trainable=True

    model=Sequential()
    model.add(_base_model)

    model.add(Dropout(rate=0.5))
    model.add(Dense(model_config.NUM_CLASS, activation='softmax', name="output"))

    # Optimizer
    if model_config.LR:
        adam=Adam(lr=model_config.LR)
    else:
        adam=Adam()

    # Compile
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Summary
    model.summary()
    
    return model

def remove_padding(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def show_im(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


def crop_or_pad(image, size=(299, 299)) :
        
    image = image.astype(np.float32)
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    image_max = max(h, w)
    scale = float(min(size)) / image_max

    image = cv2.resize(image, (int(w * scale), int(h * scale)))

    h, w = image.shape[:2]
    top_pad = (size[1] - h) // 2
    bottom_pad = size[1] - h - top_pad
    left_pad = (size[0] - w) // 2
    right_pad = size[0] - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    
    # Fix image normalization
    if np.nanmax(image) > 1 :
        image = np.divide(image, 255.)

    return image
    
def plot_cm(cm, title, yticks, xticks, cal_type="precision", save_dir=None):
    
    PLT_SIZE = 10 + int(20. / float(len(xticks)))
    
    plt.rc('font', size=PLT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=PLT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=PLT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=PLT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=PLT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=PLT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=PLT_SIZE)  # fontsize of the figure title
    
    keep_cm = cm
    if cal_type == 'recall':
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        
    else:
        cm_sum = cm.sum(axis=0)[:, np.newaxis]
        
    cm = cm.astype('float') / cm_sum
        
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)

    plt.xticks(np.arange(len(xticks)), xticks, rotation=0, fontproperties=fontprop)
    plt.yticks(np.arange(len(yticks)), yticks, rotation=0, fontproperties=fontprop)
 
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        plt.text(j, i-0.1, format(cm[i, j], '.2f'),
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
        plt.text(j, i+0.1, format(keep_cm[i, j], 'd') + " / " + str(cm_sum[i][0]),
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
    #plt.tight_layout()
    plt.ylabel('Ground_Truth')
    plt.xlabel('Prediction')
    if save_dir: plt.savefig(save_dir)
    plt.show()
    plt.close()
    
    
def get_model(model_path):
    json_file = open(model_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()

    return model_from_json(loaded_model_json)