from model_training import Model
import numpy as npy
import os
import tkinter as tk
from tkinter import filedialog

import keras.utils
from sklearn.metrics import accuracy_score
from keras.src.backend.common.global_state import clear_session
from keras.models import Sequential
from keras.layers import Dense, LSTM

class EvalModel(Model):

    def __init__(self, data_folder):
        super().__init__(data_folder)
    
    def loadWeights(self):
        root = tk.Tk()
        root.withdraw()
        filePath = filedialog.askopenfilename(title="Please choose the model to evaluate")
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

def setTestingDir():
    root = tk.Tk()
    root.withdraw()
    trainingDir = filedialog.askdirectory(title="Select the folder containing the testing data")
    return trainingDir

if __name__ == '__main__':
    data_path = setTestingDir()
    testModel = EvalModel(data_path)
    print(testModel.evaluate())





