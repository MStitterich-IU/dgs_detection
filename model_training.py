import keras.utils
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Dense, LSTM, Input
from keras.models import Sequential
from keras.src.backend.common.global_state import clear_session
import numpy as npy
import os
import tkinter as tk
from tkinter import filedialog, simpledialog


class Model():

    def __init__(self, data_folder):
        self.data_path = data_folder
        self.gestures = os.listdir(self.data_path)
        self.output_count = npy.array(self.gestures).shape[0]

        self.model = Sequential()
        self.model.add(Input(shape=(30,1662)))
        self.model.add(LSTM(64, return_sequences=True, activation='relu'))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.output_count, activation='softmax'))
    
    def get_gesture_count(self):
        root = tk.Tk()
        root.withdraw()
        gesture_count = simpledialog.askinteger("Gesture Count", "How many gestures can the model detect?")
        return gesture_count
        
    def load_data(self):
        self.videos, self.labels = [], []
        frames_per_video = len(os.listdir((os.path.join(self.data_path, self.gestures[0], str(1)))))
        label_mapping = {label:num for num, label in enumerate(self.gestures)}
        for gesture in self.gestures:
            for video in npy.array(os.listdir(os.path.join(self.data_path, gesture))).astype(int):
                frames = []
                for frame_nr in range(1, frames_per_video+1):
                    res = npy.load(os.path.join(self.data_path, gesture, str(video), "{}.npy".format(frame_nr)))
                    frames.append(res)
                self.videos.append(frames)
                self.labels.append(label_mapping[gesture])

    def train(self):
        self.load_data()
        x = npy.array(self.videos)
        y = keras.utils.to_categorical(self.labels).astype(int)

        log_dir = os.path.join('Logs')
        tensor_board_cb = TensorBoard(log_dir=log_dir)
        early_stop_cb = EarlyStopping("loss", patience=3, start_from_epoch=20)

        self.model.compile(optimizer='Adam', metrics=['categorical_accuracy'], loss='categorical_crossentropy')
        return self.model.fit(x, y, epochs=100, callbacks=[tensor_board_cb, early_stop_cb])

    def save_weights(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(title="Save model weights as", defaultextension='.keras')
        self.model.save(file_path)

def get_user_input():
    root = tk.Tk()
    root.withdraw()
    training_dir = filedialog.askdirectory(title="Select the folder containing the training data")
    iter_count = simpledialog.askinteger("Iteration Count", "How many training iterations do you want to run?")
    model_dir = filedialog.askdirectory(title="Select the folder for storing the trained models")
    return training_dir, iter_count, model_dir

if __name__ == '__main__':
    training_dir, iter_count, model_dir = get_user_input()
    for i in range(1,iter_count+1):
        clear_session()
        new_model = Model(data_folder=training_dir)
        history = new_model.train()
        with open(os.path.join(model_dir, 'trainingHistory.txt'), 'a') as training_history:
            training_history.write('Iteration {}: {} Epochs\n'.format(i, len(history.history['loss'])))
        new_model.model.save(os.path.join(model_dir, 'model{}.keras'.format(i)))
