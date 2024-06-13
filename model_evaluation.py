import numpy as npy
import os
import tkinter as tk
from tkinter import filedialog

import keras.utils
from sklearn.metrics import accuracy_score
from keras.src.backend.common.global_state import clear_session
from keras.models import Sequential
from keras.layers import Dense, LSTM

class Model():

    def __init__(self, data_folder):
        self.data_path = os.path.join(data_folder)
        self.gestures = os.listdir(os.path.join(self.data_path))

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(npy.array(self.gestures).shape[0], activation='softmax'))
    
    def loadData(self):
        self.videos, self.labels = [], []
        framesPerVideo = len(os.listdir((os.path.join(self.data_path, self.gestures[0], str(1)))))
        labelMapping = {label:num for num, label in enumerate(self.gestures)}
        for gesture in self.gestures:
            for video in npy.array(os.listdir(os.path.join(self.data_path, gesture))).astype(int):
                frames = []
                for frameNr in range(1, framesPerVideo+1):
                    res = npy.load(os.path.join(self.data_path, gesture, str(video), "{}.npy".format(frameNr)))
                    frames.append(res)
                self.videos.append(frames)
                self.labels.append(labelMapping[gesture])
    
    def loadWeights(self):
        root = tk.Tk()
        root.withdraw()
        filePath = filedialog.askopenfilename(title="Please select the model's file")
        self.model.load_weights(filePath)

    def evaluate(self):
        self.loadWeights()
        self.loadData()
        x = npy.array(self.videos)
        y = keras.utils.to_categorical(self.labels).astype(int)
        ypred = self.model.predict(x)
        ytrue = npy.argmax(y, axis=1).tolist()
        ypred = npy.argmax(ypred, axis=1).tolist()
        return accuracy_score(ytrue, ypred)

if __name__ == '__main__':

    testModel = Model('testing_data')
    print(testModel.evaluate())





