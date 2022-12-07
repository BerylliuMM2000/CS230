import mymodels
import augment
import plot_confusion_matrix
import sys
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt

if len(sys.argv) < 2:
    print("Usage: ")
    print("   $python3 hyperparameter_tuning.py  <model.h5> <image_path.png>")
model_path = sys.argv[1]
image_path = sys.argv[2]

EPOCHS = 20
early_stopping = EarlyStopping(monitor = 'val_acc',patience = 5,restore_best_weights=True)
checkpoint_cb = ModelCheckpoint(model_path, save_best_only=True)

# Example of tuning ResNet 50 with adam optimizer: 
tuner = kt.RandomSearch(mymodels.tune_resnet50_adam,
                     objective='val_acc',
                     overwrite = False,
                     max_trials = 100)
tuner.search(augment.train_images_aug, augment.train_labels_aug, epochs = 15, 
validation_data = (augment.valid_images, augment.valid_labels))
# Output the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)
# Best model, fit and save to give path
resnet50 = tuner.get_best_models()[0]
history = resnet50.fit(
        augment.train_images_aug, augment.train_labels_aug,
        epochs           = EPOCHS,
        validation_data  = (augment.valid_images, augment.valid_labels),
        verbose          = 1,
        callbacks        = [checkpoint_cb, early_stopping],)
# get class predictions on validation set
y_prob = resnet50.predict(augment.valid_images)
y_pred = y_prob.argmax(axis=-1)

# get actual classes
y_actual = np.argmax(augment.valid_labels, axis=-1)

# plot training metrics
plot_confusion_matrix.plot_training_metrics(history,resnet50,augment.valid_images,augment.valid_labels,y_actual,y_pred,['mild','moderate','normal','very-mild'])

