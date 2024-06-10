import numpy as npy
import os
import project_setup as pSetup

import keras.utils
from sklearn.model_selection import train_test_split
from keras.src.backend.common.global_state import clear_session
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import TensorBoard, EarlyStopping


gestures = os.listdir(os.path.join(pSetup.DATA_PATH))
framesPerVideo = len(os.listdir((os.path.join(pSetup.DATA_PATH, gestures[0], str(1)))))
labelMapping = {label:num for num, label in enumerate(gestures)}

#Loading previously recorded data
videos, labels = [], []
for gesture in gestures:
    for video in npy.array(os.listdir(os.path.join(pSetup.DATA_PATH, gesture))).astype(int):
        frames = []
        for frameNr in range(1, framesPerVideo+1):
            res = npy.load(os.path.join(pSetup.DATA_PATH, gesture, str(video), "{}.npy".format(frameNr)))
            frames.append(res)
        videos.append(frames)
        labels.append(labelMapping[gesture])

x = npy.array(videos)
y = keras.utils.to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

log_dir = os.path.join('Logs')
tensorBoardCB = TensorBoard(log_dir=log_dir)
earlyStopCB = EarlyStopping("val_loss", patience=5, start_from_epoch=50)



#clear_session()
keras.utils.set_random_seed(1)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(npy.array(gestures).shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=100, callbacks=[tensorBoardCB, earlyStopCB])

model.save('gesture_detection_model.keras')
