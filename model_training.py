import numpy as npy
import os
import tkinter as tk
from tkinter import filedialog

import keras.utils
from keras.src.backend.common.global_state import clear_session
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.callbacks import TensorBoard, EarlyStopping

class Model():

    def __init__(self, data_folder):
        self.data_path = data_folder
        self.gestures = os.listdir(self.data_path)

        self.model = Sequential()
        self.model.add(Input(shape=(30,1662)))
        self.model.add(LSTM(64, return_sequences=True, activation='relu'))
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

    def train(self):
        self.loadData()
        x = npy.array(self.videos)
        y = keras.utils.to_categorical(self.labels).astype(int)

        log_dir = os.path.join('Logs')
        tensorBoardCB = TensorBoard(log_dir=log_dir)
        earlyStopCB = EarlyStopping("loss", patience=3, start_from_epoch=20)

        self.model.compile(optimizer='Adam', metrics=['categorical_accuracy'], loss='categorical_crossentropy')
        return self.model.fit(x, y, epochs=100, callbacks=[tensorBoardCB, earlyStopCB])

    def saveWeights(self):
        root = tk.Tk()
        root.withdraw()
        filePath = filedialog.asksaveasfilename(title="Save model weights as", defaultextension='.keras')
        self.model.save(filePath)

def setTrainingDir():
    root = tk.Tk()
    root.withdraw()
    trainingDir = filedialog.askdirectory(title="Wählen Sie den Ordner mit den Trainingsdaten aus")
    return trainingDir

if __name__ == '__main__':
    trainingDir = setTrainingDir()
    modelPath = os.path.join(os.getcwd(), 'models', 'leadingAnim_noHands', 'multi_cam')
    for i in range(32,41):
        clear_session()
        newModel = Model(trainingDir)
        history = newModel.train()
        with open(os.path.join(modelPath, 'trainingHistory.txt'), 'a') as trainingHistory:
            trainingHistory.write('Iteration {}: {} Epochs\n'.format(i, len(history.history['loss'])))
        newModel.model.save(os.path.join(modelPath, 'model{}.keras'.format(i)))
