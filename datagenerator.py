from PIL import Image
from os.path import join
from math import floor
import glob
import os
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#gather dataset
def create_df_dataset(test_size = 0.25):
    """returns train and test set split in panda df format along with trian and test set size:
    [train_df, tesst_df, train_size, test_size]
    """
    cwd = os.getcwd()
    image_path = join(cwd,'./nailgun')
    good_images_path = join(image_path, 'good')
    bad_images_path = join(image_path,'bad')
    good_images = glob.glob(join(good_images_path,'*'))
    good_labels = len(good_images) * [1]
    bad_images = glob.glob(join(bad_images_path,'*'))
    bad_labels = len(bad_images) * [0]

    all_images = good_images + bad_images
    all_labels = good_labels + bad_labels 

    df = pd.DataFrame(list(zip(all_images, all_labels)), 
               columns =['images', 'labels']) 

    
    df = df.sample(frac = 1) #shuffle
    train, test = train_test_split(df, test_size=test_size)
    return train,test,train.shape[0],test.shape[0]

def create_tf_image_generator(df_train,df_test, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE):
    train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
    train_data_gen = train_image_generator.flow_from_dataframe(dataframe = df_train,
        directory=".\train_imgs",
        x_col='images',
        y_col = 'labels',
        target_size= (IMAGE_HEIGHT, IMAGE_WIDTH),
        colormode = 'grayscale',
        batch_size = BATCH_SIZE,
        class_mode='raw')
    
    val_data_gen = train_image_generator.flow_from_dataframe(dataframe = df_test,
        directory=".\val_imgs",
        x_col='images',
        y_col = 'labels',
        target_size= (IMAGE_HEIGHT,IMAGE_WIDTH),
        colormode = 'grayscale',
        batch_size = BATCH_SIZE,
        class_mode='raw')

    return train_data_gen, val_data_gen
