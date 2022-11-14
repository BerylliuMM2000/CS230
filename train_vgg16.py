import tensorflow as tf
import models
import data_generator
import matplotlib.pyplot as plt
import sys

model_save_path = sys.argv[1]
fig_save_path = sys.argv[2]
# Optional third argument indicates transfer learning or not
transfer = False
if len(sys.argv) > 3:
    transfer = True
METRICS = [tf.keras.metrics.AUC(name='auc'), "acc"]

def train_model(model_save_path, fig_save_path, transfer):
    # Train all parameter from scratch
    if transfer:
        baseline_vgg16_model = models.build_VGG16_transfer()
    else:
        baseline_vgg16_model = models.build_VGG16_full()
    train_ds, val_ds = data_generator.train_ds, data_generator.val_ds
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(models.exponential_decay(0.01, 20))
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    baseline_vgg16_model.compile(optimizer='adam',
                            loss=tf.losses.CategoricalCrossentropy(), 
                            metrics=METRICS)
    history = baseline_vgg16_model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    epochs=models.EPOCHS)

    # Plot the history of training AUC and loss
    fig, ax = plt.subplots(1, 3, figsize=(20, 3))
    ax = ax.ravel()
    for i, met in enumerate(['auc', 'loss', 'acc']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train', 'val'])
    plt.savefig(fig_save_path)
    return baseline_vgg16_model

train_model(model_save_path, fig_save_path, transfer)