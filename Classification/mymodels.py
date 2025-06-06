import tensorflow as tf
from tensorflow.keras import backend, models, layers, Sequential
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, Flatten, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend, models, layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121,InceptionV3, Xception, ResNet101
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import keras_tuner as kt
from keras.optimizers import SGD, Adam

BATCH_SIZE = 16
IMAGE_SIZE = [176, 208]
EPOCHS = 100
NUM_CLASSES = 4
METRICS = [tf.keras.metrics.AUC(name='auc'), "acc"]

""" Two types of models are present here. The functions starting with 'tune', also with
    'hp' as the only argument return models only for hyperparameter tuning. Other functions 
    return normal models. The default argument values come from hyperparameter tuning. One 
    should not change these values when initiating and fitting models unless a better tuner
    yield better results.
"""

def tune_resnet50_adam(hp):
    """ Build a Resnet-50 model for hyperparameter tuning

    This model is a modification of the original Resnet-50 model structure.
    It inherits all the convolution layers, but the dense and softmax layers
    are rebuild. 2 FC layers are used. The size of FC layers will be tuned 
    upon hyperparameter tuning. All parameters will be retrained upon training.

    Solver Adam is used for this model

    Tuning hyperparameters:                               Best value:
    unit1: The size of the first FC layer                 512
    unit2: The size of the second FC layer                128
    lr: learning rate of Adam() solver                    2.25e-4
    beta1: the beta_1 parameter of Adam() solver          0.963
    beta2: the beta_2 parameter of Adam() solver          0.9997
    """
    model = Sequential()
    model.add(tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False))
    model.add(AveragePooling2D())
    model.add(Flatten())

    dense1 = hp.Int('unit1', min_value=512, max_value=2048, step=512)
    dense2 = hp.Int('unit2', min_value=128, max_value=512, step=128)

    model.add(Dense(dense1))
    model.add(Dense(dense2))
    model.add(Dense(4, activation='softmax'))

    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_beta_1 = hp.Float("beta1", min_value = 0.9, max_value = 0.98, sampling = "linear")
    hp_beta_2 = hp.Float("beta2", min_value = 0.99, max_value = 0.9999, sampling = "log")
    adam = Adam(learning_rate = hp_learning_rate, beta_1=hp_beta_1, beta_2=hp_beta_2)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=METRICS)
    return model

def tune_resnet50_sgd(hp):
    """ Build a Resnet-50 model for hyperparameter tuning

    This model is a modification of the original Resnet-50 model structure.
    It inherits all the convolution layers, but the dense and softmax layers
    are rebuild. 2 FC layers are used. The size of FC layers will be tuned 
    upon hyperparameter tuning. All parameters will be retrained upon training.

    Solver SGD is used for this model

    Tuning hyperparameters:                               Best value:
    unit1: The size of the first FC layer                 1280
    unit2: The size of the second FC layer                256
    lr: learning rate of Adam() solver                    2.7e-3
    momentum: the momentum parameter of SGD() solver      0.95
    """
    model = Sequential()
    model.add(tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False))
    model.add(AveragePooling2D())
    model.add(Flatten())

    dense1 = hp.Int('unit1', min_value=256, max_value=2048, step=256)
    dense2 = hp.Int('unit2', min_value=32, max_value=256, step=32)

    model.add(Dense(dense1))
    model.add(Dense(dense2))
    model.add(Dense(4, activation='softmax'))

    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_momentum = hp.Choice('momentum', values=[0.9, 0.95, 0.97])
    sgd = SGD(learning_rate=hp_learning_rate, decay=1e-6, momentum=hp_momentum, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=METRICS)
    return model

def build_resnet50_sgd(unit1=1280, unit2=256, lr=2.7e-3, momentum=0.95):
    """ Build a modified ResNet-50 model using the best hyperparameters.

    This model is a modification of the original Resnet-50 model structure.
    It inherits all the convolution layers, but the dense and softmax layers
    are rebuild. 2 FC layers are used.

    SGD is used as optimizer.
    """
    model = Sequential()
    model.add(tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(unit1))
    model.add(Dense(unit2))
    model.add(Dense(4, activation='softmax'))
    sgd = SGD(learning_rate=lr, decay=1e-6, momentum=momentum, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=METRICS)
    return model

def build_resnet50_adam(unit1=512, unit2=128, lr=2.25e-4, beta1=0.963, beta2=0.9997):
    """ Build a modified ResNet-50 model using the best hyperparameters.

    This model is a modification of the original Resnet-50 model structure.
    It inherits all the convolution layers, but the dense and softmax layers
    are rebuild. 2 FC layers are used.

    Adam is used as the optimizer
    """
    model = Sequential()
    model.add(tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(unit1))
    model.add(Dense(unit2))
    model.add(Dense(4, activation='softmax'))
    adam = Adam(learning_rate = lr, beta_1=beta1, beta_2=beta2)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=METRICS)
    return model


# TODO: baseline model of inception and vgg16
def build_vgg16():
    ''' VGG-16 Baseline model, tune all parameters
    '''
    vgg16 = tf.keras.applications.VGG16(
        input_shape=(*IMAGE_SIZE, 3), 
        weights='imagenet', 
        include_top=False)
    # Train all parameters
    x = tf.keras.layers.Flatten()(vgg16.output)
    prediction = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=vgg16.input, outputs=prediction)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=METRICS)
    return model

def build_inception(dropout,dense_node):
    """
    Build and compile an inceptionV3 model with dense layer rebuilt
    Input: dropout rate, the number of filters in the dense node, 
    """
    model = Sequential()
    model.add(InceptionV3(weights = 'imagenet', include_top = False, input_shape = (*IMAGE_SIZE, 3)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(dense_node,activation='relu'))
    model.add(Dense(4,activation='softmax'))
    # complile the model
    model.compile(
        optimizer = 'adam',
        loss      = 'categorical_crossentropy', 
        metrics   = METRICS)
    return model
