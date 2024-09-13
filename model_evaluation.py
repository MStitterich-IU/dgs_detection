import keras.utils
from model_training import Model
import numpy as npy
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog


class EvalModel(Model):

    def __init__(self, data_folder):
        super().__init__(data_folder)
    
    def load_weights(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Please choose the model to evaluate")
        self.model.load_weights(file_path)

    def evaluate(self):
        self.load_weights()
        self.load_data()
        x = npy.array(self.videos)
        y = keras.utils.to_categorical(self.labels).astype(int)
        ypred = self.model.predict(x)
        ytrue = npy.argmax(y, axis=1).tolist()
        ypred = npy.argmax(ypred, axis=1).tolist()
        return accuracy_score(ytrue, ypred)

def set_testing_dir():
    root = tk.Tk()
    root.withdraw()
    training_dir = filedialog.askdirectory(title="Select the folder containing the testing data")
    return training_dir

if __name__ == '__main__':
    data_path = set_testing_dir()
    test_model = EvalModel(data_path)
    print(test_model.evaluate())





