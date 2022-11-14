import data_generator
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

METRICS = [tf.keras.metrics.AUC(name='auc'), 'acc']

model_path = sys.argv[1]
test_ds = data_generator.test_ds

# Load all parameters from existing model
baseline_model = tf.keras.models.load_model(model_path)
baseline_model.compile(optimizer='adam',
                        loss=tf.losses.CategoricalCrossentropy(), 
                        metrics=METRICS)

_ = baseline_model.evaluate(test_ds)