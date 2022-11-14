import tensorflow as tf
import data_generator

BATCH_SIZE = 16
IMAGE_SIZE = [224, 224]
EPOCHS = 100
NUM_CLASSES = data_generator.NUM_CLASSES

def conv_block(filters, kernel=3):
    ''' One convolutional block contains one convolutional layer and a pooling layer

    The convolutional layer has kernel size 3*3 by default, and applies a same padding.
    Each pooling layer average pools and reduces the input size by half.
    '''
    block = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])  
    return block

def dense_block(units, dropout_rate = 0):
    ''' Dense blocks for the last fewer FC layers
    '''
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block

def build_lenet5_baseline():
    ''' Implement a baseline model similar to the Lenet-5 structure

    Contains 3 conv_blocks, a flatten layer, 2 FC layers, and then a softmax layer

    Return: tf.keras.Sequential() baseline model
    '''
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
        conv_block(6),
        conv_block(16),
        conv_block(32),
        tf.keras.layers.Flatten(),
        dense_block(320),
        dense_block(80),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def build_VGG16_transfer():
    vgg16 = tf.keras.applications.VGG16(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False)
    # Transfer learning the imagenet parameters, only train last layer
    for layer in vgg16.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(vgg16.output)
    prediction = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inputs=vgg16.input, outputs=prediction)

def build_VGG16_full():
    vgg16 = tf.keras.applications.VGG16(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False)
    # Train all parameters
    x = tf.keras.layers.Flatten()(vgg16.output)
    prediction = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inputs=vgg16.input, outputs=prediction)

def build_resnet50_transfer():
    resnet50 = tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False)
    # Transfer learning the imagenet parameters, only train last layer
    for layer in resnet50.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(resnet50.output)
    prediction = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inputs=resnet50.input, outputs=prediction)

def build_resnet50_full():
    resnet50 = tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False)
    # Train all parameters
    x = tf.keras.layers.Flatten()(resnet50.output)
    prediction = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inputs=resnet50.input, outputs=prediction)    

def exponential_decay(lr0, s):
    return lambda epoch: lr0 * 0.1 ** (epoch / s)
