from tensorflow import keras

# The encoder learns vector representations of the MRI scans
def create_encoder():
    resnet = keras.applications.ResNet50V2(
    #resnet = keras.applications.ResNet101V2(
    #resnet = keras.applications.EfficientNetB2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg")

    model = keras.Model(inputs=keras.Input(shape=input_shape), outputs=resnet(data_augmentation(inputs)))
    return model

#lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
#    initial_learning_rate=learning_rate, decay_steps=10)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( # added on 12/5 afternoon
    initial_learning_rate = 0.01,
    decay_steps = 1000,
    decay_rate = 0.95
)

# Add fully-conencted layers on the encoder
def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    #features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dense(hidden_units, activation="relu")(features) 
    features = layers.Dense(hidden_units, activation="relu")(features) 
    #features = layers.BatchNormalization()(features)  
    #features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(4, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        #optimizer = keras.optimizers.Adam(learning_rate), # the original code
        optimizer = keras.optimizers.SGD(learning_rate),#, decay=1e-6,momentum=0.5)
        #optimizer = keras.optimizers.SGD(learning_rate=lr_schedule),
        # optimizer = keras.experimental.CosineDecay(learning_rate, decay_steps=10),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]#, keras.metrics.AUC(name='auc')]
    )
    return model
