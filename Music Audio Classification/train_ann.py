# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:49:09 2020

@author: nam
"""

# Import the libraries
import feature_engineer as fe
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
tf.debugging.set_log_device_placement(True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)


# Data Preprocessing
sc = fe.init_scaler()
X, Y = fe.preprocessing_csv('audio_data.csv', sc)


# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)


# Build ANN Model
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 256, activation = 'relu', input_shape = (X_train.shape[1],)))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 64, activation = 'relu'))
    classifier.add(Dense(units = 10, activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 128, epochs = 200)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)


cm = confusion_matrix(y_test, predictions)

print('Classification Report on Test set')
print(classification_report(y_test, predictions))


# Evaluate ANN with CV
cv_classifier = KerasClassifier(build_fn = build_classifier, batch_size = 128, epochs = 200)
cv_accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
cv_mean = cv_accuracies.mean()


# Predict a single audio file
audio = fe.extract_features_id(999)
X_audio, _ = fe.preprocessing_df(audio, sc)

X_predict = classifier.predict(X_audio)