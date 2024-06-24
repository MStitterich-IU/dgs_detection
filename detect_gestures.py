from record_gestures import DataRecorder
from model_training import Model
import cv2
import numpy as npy
import os
import tkinter as tk
from tkinter import filedialog

class GesturePrediction(DataRecorder, Model):

    def __init__(self):
        DataRecorder.__init__(self, 'training_data')
        Model.__init__(self, os.path.join('training_data', 'training_data_leadingAnim_noHands'))

    def load_model_weights(self):
        root = tk.Tk()
        root.withdraw()
        filePath = filedialog.askopenfilename(title="Please select the model's file")
        self.model.load_weights(filePath)
    
    def record_gestures(self):
        sequence = []
        predictionAccuracy = 0
        threshold = 0.7
        self.load_model_weights()
        #Start recording process
        with self.holisticModel.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hModel:
            while self.cameraCapture.isOpened():

                # Read incoming camera frames
                ret, frame = self.cameraCapture.read()

                #Detect and process gesture keypoints
                frame, results = self.keypoint_detection(frame, hModel)

                #Read landmark values
                lmarkValues = self.read_lmark_values(results)
                sequence.append(lmarkValues)

                cv2.rectangle(frame, (0, 550), (800, 600), (255, 0, 0), -1)
                
                #Predict gesture after 30 recorded frames
                if len(sequence) == 30:
                    res = self.model.predict(npy.expand_dims(sequence, axis=0))[0]
                    #predictions.append(npy.argmax(res))
                    #predictions = predictions [-5:]
                    predictionAccuracy = res[npy.argmax(res)]
                    if predictionAccuracy < threshold:
                        sequence = []
                        continue
                    #Make sure that prediction is not just a fluke
                    #if npy.unique(predictions[-2:])[0] == npy.argmax(res):
                    screenText = "Geste: " + self.gestures[npy.argmax(res)] + " Genauigkeit: " + str(predictionAccuracy)
                    
                    cv2.putText(frame, screenText, (10, 585), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    cv2.imshow('Detecting gestures', frame)
                    cv2.waitKey(1000)
                    sequence = []
                    continue
                
                


                cv2.imshow('Detecting gestures', frame)

                # Quit capture gracefully by pressing ESC
                key = cv2.waitKey(10)
                if key == 27:
                    self.cameraCapture.release()
                    cv2.destroyAllWindows()
                    return

            


if __name__ == '__main__':
    prediction = GesturePrediction()
    prediction.record_gestures()

