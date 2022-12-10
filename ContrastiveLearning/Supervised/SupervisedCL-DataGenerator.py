import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from sklearn.model_selection import train_test_split
import glob


input_shape = (208,208,1)

AUTOTUNE = tf.data.experimental.AUTOTUNE
 

MildDemented_train = glob.glob('./Alzheimer_s Dataset/train/MildDemented/*.*')
ModeratedDemented_train = glob.glob('./Alzheimer_s Dataset/train/ModerateDemented/*.*')
NonDemented_train = glob.glob('./Alzheimer_s Dataset/train/NonDemented/*.*')
VeryMildDemented_train = glob.glob('./Alzheimer_s Dataset/train/VeryMildDemented/*.*')

MildDemented_test = glob.glob('./Alzheimer_s Dataset/test/MildDemented/*.*')
ModeratedDemented_test = glob.glob('./Alzheimer_s Dataset/test/ModerateDemented/*.*')
NonDemented_test = glob.glob('./Alzheimer_s Dataset/test/NonDemented/*.*')
VeryMildDemented_test = glob.glob('./Alzheimer_s Dataset/test/VeryMildDemented/*.*')


train_data = []
train_labels = []
test_data = []
test_labels = []

from tqdm import tqdm

for i in tqdm(MildDemented_train):   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
    target_size= (208,208))
    image=np.array(image)
    train_data.append(image)
    train_labels.append(0)
    classes = ['MildDemented']
    
for i in tqdm(ModeratedDemented_train):   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
    target_size= (208,208))
    image=np.array(image)
    train_data.append(image)
    train_labels.append(1)    

for i in tqdm(NonDemented_train):   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
    target_size= (208,208))
    image=np.array(image)
    train_data.append(image)
    train_labels.append(2)

for i in tqdm(VeryMildDemented_train):   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
    target_size= (208,208))
    image=np.array(image)
    train_data.append(image)
    train_labels.append(3)


for i in tqdm(MildDemented_test):   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
    target_size= (208,208))
    image=np.array(image)
    test_data.append(image)
    test_labels.append(0)
    
for i in tqdm(ModeratedDemented_test):   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
    target_size= (208,208))
    image=np.array(image)
    test_data.append(image)
    test_labels.append(1)

for i in tqdm(NonDemented_test):   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
    target_size= (208,208))
    image=np.array(image)
    test_data.append(image)
    test_labels.append(2)

for i in tqdm(VeryMildDemented_test):   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
    target_size= (208,208))
    image=np.array(image)
    test_data.append(image)
    test_labels.append(3)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,random_state=42)
