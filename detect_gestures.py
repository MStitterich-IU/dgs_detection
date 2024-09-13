from record_gestures import DataRecorder
from model_training import Model
import cv2
import numpy as npy
import os
import tkinter as tk
from tkinter import filedialog

class GesturePrediction(DataRecorder, Model):

    def __init__(self):
        DataRecorder.__init__(self)
        Model.__init__(self, self.load_gestures())

    def load_model_weights(self):
        root = tk.Tk()
        root.withdraw()
        filePath = filedialog.askopenfilename(title="Please select the model's file")
        self.model.load_weights(filePath)
    
    def load_gestures(self):
        root = tk.Tk()
        root.withdraw()
        gesturesPath = filedialog.askdirectory(title="Please select the folder containing the training data")
        return gesturesPath
    
    def visualize_lmarks(self, frame, results, camNr=1):
        #Drawing specs for changing the  presentation style
        POSE_LMARK = self.mpDrawUtil.DrawingSpec(color=(168,52,50), thickness=2, circle_radius=3)
        POSE_CONN = self.mpDrawUtil.DrawingSpec(color=(50,50,168), thickness=2, circle_radius=2)
        HAND_LMARK = self.mpDrawUtil.DrawingSpec(color=(134,109,29), thickness=2, circle_radius=3)
        HAND_CONN = self.mpDrawUtil.DrawingSpec(color=(161,161,160), thickness=2, circle_radius=2)

        # Visualize face contours
        #mpDrawUtil.draw_landmarks(frame, results.face_landmarks, holisticModel.FACEMESH_CONTOURS)
        
        # Posture landmarks
        self.mpDrawUtil.draw_landmarks(frame, results.pose_landmarks, self.holisticModel.POSE_CONNECTIONS, POSE_LMARK, POSE_CONN) 
        # Left hand connections
        self.mpDrawUtil.draw_landmarks(frame, results.left_hand_landmarks, self.holisticModel.HAND_CONNECTIONS, HAND_LMARK, HAND_CONN) 
        # Right hand connections
        self.mpDrawUtil.draw_landmarks(frame, results.right_hand_landmarks, self.holisticModel.HAND_CONNECTIONS, HAND_LMARK, HAND_CONN) 

    
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

                #Visualize keypoints and connections
                self.visualize_lmarks(frame, results)

                #Read landmark values
                lmarkValues = self.read_lmark_values(results)
                sequence.append(lmarkValues)

                cv2.rectangle(frame, (0, 550), (800, 600), (255, 0, 0), -1)
                
                #Predict gesture after 30 recorded frames
                if len(sequence) == 30:
                    res = self.model.predict(npy.expand_dims(sequence, axis=0))[0]
                    predictionAccuracy = res[npy.argmax(res)]
                    if predictionAccuracy < threshold:
                        sequence = []
                        continue
                    #Make sure that prediction is not just a fluke
                    #if npy.unique(predictions[-2:])[0] == npy.argmax(res):
                    screenText = "Gesture: " + self.gestures[npy.argmax(res)] + " Accuracy: " + str("%.2f" % predictionAccuracy)
                    
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

