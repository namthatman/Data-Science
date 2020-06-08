# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:59:13 2020

@author: nam
"""

# Import the libraries
import pandas as pd
import numpy as np
import librosa
import os
import pathlib
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing import image



def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


# Extract the spectrogram on audio files
def extract_spectrogram_all():
    genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    for genre in genres:
        pathlib.Path(f'spectrogram/{genre}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'genres/{genre}'):
            songname = f'genres/{genre}/{filename}'
            y, rate = librosa.load(songname, duration=15)
            plt.figure(figsize=(8.64,5.76))
            plt.axis('off') # no axis
            plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            S = librosa.feature.melspectrogram(y=y, sr=rate, n_fft=2048)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            plt.savefig(f'spectrogram/{genre}/{filename[:-3].replace(".", "")}.png', bbox_inches=None, pad_inches=0)
            plt.clf()
            plt.close()
            
            
# Extract the spectrogram on 1 audio file
def extract_spectrogram_id(id):
    path,filename,genre = get_filename(id)
    pathlib.Path(f'id_spectrogram/{genre}').mkdir(parents=True, exist_ok=True)      
    y, rate = librosa.load(path, duration=15)
    
    plt.figure(figsize=(8.64,5.76))
    plt.axis('off') # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=y, sr=rate, n_fft=2048)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(f'id_spectrogram/{genre}/{filename[:-3].replace(".", "")}.png', bbox_inches=None, pad_inches=0)
    plt.clf()
    plt.close()
    

# Extract features from all audio files, return dataframe
def extract_features_all():
    genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    append_list = []
    for genre in genres:
        for filename in os.listdir(f'genres/{genre}'):
            path = f'genres/{genre}/{filename}'
            y, rate = librosa.load(path, mono=True, duration=30)
            rmse = librosa.feature.rms(y=y)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=rate)
            centroid = librosa.feature.spectral_centroid(y=y, sr=rate)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=rate)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=rate)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=rate)
            
            to_append = [f'{filename}',f'{genre}',np.mean(rmse),np.mean(chroma_stft),
                         np.mean(centroid),np.mean(bandwidth),np.mean(rolloff),np.mean(zcr)]
            
            for i in mfcc:
                to_append.append(np.mean(i))
                
            append_list.append(to_append)
        
    df = pd.DataFrame(append_list)
    return df


# Get location of a given song by id, return path, filename, genre
def get_filename(id):
    if id > 999:
        id = 999
    if id < 0:
        id = 0
    g = int(id / 100)
    gid = str(int(id % 100))
    if len(gid) < 2:
        gid = '0' + str(gid)
    else:
        gid = str(gid)
    
    genres_dist = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    genre = genres_dist[g]
    
    path = f'genres/{genre}/{genre}.000{gid}.wav'
    filename = f'{genre}.000{gid}.wav'
    
    return path,filename,genre
    

# Extract features from a given ID song, return dataframe
def extract_features_id(id):
    append_list = []
    
    path,filename,genre = get_filename(id)    
    y, rate = librosa.load(path, mono=True, duration=30)
    
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=rate)
    centroid = librosa.feature.spectral_centroid(y=y, sr=rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=rate)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=rate)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=rate)
    
    to_append = [f'{filename}',f'{genre}',np.mean(rmse),np.mean(chroma_stft),
                 np.mean(centroid),np.mean(bandwidth),np.mean(rolloff),np.mean(zcr)]    
    for i in mfcc:
        to_append.append(np.mean(i))
        
    append_list.append(to_append)
    
    df = pd.DataFrame(append_list)
    return df


# Save data into csv
def save_csv(df):
    df.to_csv('audio_data.csv', index=False, header=False)
    

# Data Preprocessing for csv
def preprocessing_csv(csv, sc):
    data = pd.read_csv(csv, header=None)
    data = data.drop([0], axis=1)
    
    # Label encoding
    y_ = data.iloc[:,0] 
    le = LabelEncoder()
    y_ = le.fit_transform(y_)
    
    # Feature Scalling
    x_ = data.iloc[:,1:]
    x_ = sc.transform(x_)
    
    return x_, y_


# Data Preprocessing for df
def preprocessing_df(data, sc):
    data = data.drop([0], axis=1)
    
    # Label encoding
    y_ = data.iloc[:,0] 
    le = LabelEncoder()
    y_ = le.fit_transform(y_)
    
    # Feature Scalling
    x_ = data.iloc[:,1:]
    x_ = sc.transform(x_)
    
    return x_, y_


# Initiate scaler
def init_scaler():
    data = pd.read_csv('audio_data.csv', header=None)
    sc = StandardScaler()
    x_ = data.iloc[:,2:]
    x_ = sc.fit_transform(x_)
    return sc


def get_input_image(id):
    extract_spectrogram_id(id)
    path, filename, genre = get_filename(id)
    new_path = f'id_spectrogram/{genre}/{filename[:-3].replace(".", "")}.png'
    input_image = image.load_img(new_path, grayscale=False, color_mode='rgb', target_size=(64, 64), interpolation='nearest')
    input_image = image.img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis = 0)
    return input_image


def get_song_class(indice):
    genres_dist = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    song_class = genres_dist[indice]
    return song_class
    



# Main
#extract_spectrogram_all()
#extract_spectrogram_id(996)
    

#scaler = init_scaler()
#X, Y = preprocessing_csv('audio_data.csv', scaler)

#data = pd.read_csv('audio_data.csv', header=None)  
#Xdf, Ydf = preprocessing_df(data, scaler)

#test = extract_features_id(0)
#X_test, Y_test = preprocessing_df(test, scaler)
    
# Build csv
#data = extract_features_all()
#save_csv(data)
    
#extract_spectrogram_id(0)
#extract_spectrogram_id(999)
    
#extract_spectrogram_all()