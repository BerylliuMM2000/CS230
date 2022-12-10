# Deep Learning for Alzheimer's MRI Classification

Github Repository for CS230 Fall 2022 Project - Deep Learning Methods for Alzheimer Disease Prediction

## Datasets
- Datasets extracted from Kaggle, specified in `Alzheimer_s Dataset` folder

## Main Files:
- `data_generator.py`: process images from input directory into appropriate format for CNNs. (Deprecated and replaced by `augmented.py`)
- `augment.py`: Process image from directories and perform data augmentation on training data into appropriate format for CNNs. 
- `mymodels.py`: list of models used for this study.
- `train*.py`: Trains a particular model (Will merge all files into a comprehensive one later), saves model to `.h5` format.
- `hyperparameter_tuning.py`: Tunes a particular model given in `mymodels.py`, saves the best model to `.h5` format.
- `evaluate_and_plot.py`: evaluate a model (in `*.h5` format) on validation or test set, and plot confusion matrix and/or loss curves.
- `ContrastiveLearning`: Codes for the contrastive learning part of this study.
