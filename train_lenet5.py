import tensorflow as tf
import models
import data_generator
import matplotlib.pyplot as plt
import sys

model_save_path = sys.argv[1]
fig_save_path = sys.argv[2]
METRICS = [tf.keras.metrics.AUC(name='auc'), "acc"]

def train_model(model_save_path, fig_save_path, num_epoch = models.EPOCHS):
    # Train all parameter from scratch
    baseline_model = models.build_lenet5_baseline()
    train_ds, val_ds = data_generator.train_ds, data_generator.val_ds
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(models.exponential_decay(0.01, 20))
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    baseline_model.compile(optimizer='adam',
                            loss=tf.losses.CategoricalCrossentropy(), 
                            metrics=METRICS)
    history = baseline_model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    epochs=num_epoch)

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
    return baseline_model

if len(sys.argv) == 3:
    train_model(model_save_path, fig_save_path)
if len(sys.argv) == 4:
    # Epoch value specified
    train_model(model_save_path, fig_save_path, int(sys.argv[3]))
