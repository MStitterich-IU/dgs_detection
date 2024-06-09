import numpy as npy
import os
import project_setup as pSetup


actions = os.listdir(os.path.join(pSetup.DATA_PATH))
framesPerVideo = len(os.listdir((os.path.join(pSetup.DATA_PATH, actions[0], str(1)))))
labelMapping = {label:num for num, label in enumerate(actions)}

#Loading previously recorded data
videos, labels = [], []
for action in actions:
    for video in npy.array(os.listdir(os.path.join(pSetup.DATA_PATH, action))).astype(int):
        frames = []
        for frameNr in range(1, framesPerVideo+1):
            res = npy.load(os.path.join(pSetup.DATA_PATH, action, str(video), "{}.npy".format(frameNr)))
            frames.append(res)
        videos.append(frames)
        labels.append(labelMapping[action])


