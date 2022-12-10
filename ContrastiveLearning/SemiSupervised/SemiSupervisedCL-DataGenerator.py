from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 256

unlabeled_dataset_size = 3276
labeled_dataset_size = 820

image_channels = 3

# Algorithm hyperparameters
num_epochs = 50

batch_size = 50
width = 256
temperature = 0.001
contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
classification_augmentation = {"min_area": 0.75, "brightness": 0.3, "jitter": 0.1}


def prepare_dataset():
    steps_per_epoch = (unlabeled_dataset_size + labeled_dataset_size) // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
    labeled_batch_size = labeled_dataset_size // steps_per_epoch
    print(
        f"batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)"
    )

    unlabeled_train_dataset = tf.keras.utils.image_dataset_from_directory("./Try14",labels = "inferred",
        color_mode = "greyscale",batch_size = unlabeled_batch_size,shuffle=True)

    labeled_train_dataset = tf.keras.utils.image_dataset_from_directory("./Try15",labels = 'inferred',
        color_mode = "grescale",batch_size = labeled_batch_size,shuffle=True)

    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.keras.utils.image_dataset_from_directory("./Try16",labels = 'inferred',batch_size = batch_size,)
    return train_dataset, labeled_train_dataset, test_dataset
    
   
train_dataset, labeled_train_dataset, test_dataset = prepare_dataset()
