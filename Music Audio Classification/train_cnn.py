# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:44:23 2020

@author: nam
"""

# Import the libraries
import feature_engineer as fe
import pandas as pd
import numpy as np
import librosa
import split_folders
#from sklearn.metrics import roc_auc_score


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.debugging.set_log_device_placement(True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)


# Split spectrogram into train and test folders
split_folders.ratio('spectrogram/', output="img_data", seed=777, ratio=(.9,.1))


# Image Generation and Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train = train_datagen.flow_from_directory('img_data/train', 
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle = False)

test = test_datagen.flow_from_directory('img_data/val', 
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle = False)


# Build CNN model
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
classifier.add(Activation('relu'))

classifier.add(Conv2D(64, (3, 3), padding='same'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
classifier.add(Activation('relu'))

classifier.add(Conv2D(64, (3, 3), padding='same'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
classifier.add(Activation('relu'))

classifier.add(Flatten())
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 10, activation = 'softmax'))

#sgd = SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False)    
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit_generator(train,
                         steps_per_epoch = 100,
                         epochs = 100,
                         validation_data = test,
                         validation_steps = 200)


# Evaluate model
#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy',tf.keras.metrics.AUC()])
classifier.evaluate_generator(generator=test, steps=100)


# Evaluate AUC
test_label = test.classes
test_label = pd.get_dummies(test_label)
test.reset()
pred = classifier.predict_generator(test, steps=100, verbose=1)
predictions = pred[:100]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_label, predictions, average='macro', multi_class='ovr')
auc






# Save model
classifier.save('cnn_model') 


# Load model
classifier = tf.keras.models.load_model('cnn_model')

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', tf.keras.metrics.AUC(
                                                                                                    multi_label=True)])
classifier.evaluate_generator(generator=test, steps=100)


# Predict test set
test.reset()
pred = classifier.predict_generator(test, steps=100, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions = predictions[:100]
filenames = test.filenames
results = pd.DataFrame({'Filename':filenames, 'Predictions':predictions})




# Predict a single audio file
fe.extract_spectrogram_id(907)
path, filename, genre = fe.get_filename(907)

new_path = f'id_spectrogram/{genre}/{filename[:-3].replace(".", "")}.png'

test_image = image.load_img(new_path, grayscale=False, color_mode='rgb', target_size=(64, 64), interpolation='nearest')


test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result2 = classifier.predict(test_image)
result2 = np.argmax(result2,axis=1)
song_class = fe.get_song_class(result2[0])





