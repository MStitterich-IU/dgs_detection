import cv2
from model_training import Model
import numpy as npy
from record_gestures import DataRecorder
import tkinter as tk
from tkinter import filedialog


class GesturePrediction(DataRecorder, Model):

    def __init__(self):
        DataRecorder.__init__(self)
        Model.__init__(self, self.load_gestures())

    def load_model_weights(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Please select the model's file")
        self.model.load_weights(file_path)
    
    def load_gestures(self):
        root = tk.Tk()
        root.withdraw()
        gestures_path = filedialog.askdirectory(title="Please select the folder containing the training data")
        return gestures_path
    
    def visualize_lmarks(self, frame, results):
        #Drawing specs for changing the  presentation style
        POSE_LMARK = self.mp_draw_util.DrawingSpec(color=(168,52,50), thickness=2, circle_radius=3)
        POSE_CONN = self.mp_draw_util.DrawingSpec(color=(50,50,168), thickness=2, circle_radius=2)
        HAND_LMARK = self.mp_draw_util.DrawingSpec(color=(134,109,29), thickness=2, circle_radius=3)
        HAND_CONN = self.mp_draw_util.DrawingSpec(color=(161,161,160), thickness=2, circle_radius=2)

        # Visualize face contours
        #mpDrawUtil.draw_landmarks(frame, results.face_landmarks, holisticModel.FACEMESH_CONTOURS)
        
        # Posture landmarks
        self.mp_draw_util.draw_landmarks(frame, results.pose_landmarks, self.holistic_model.POSE_CONNECTIONS, POSE_LMARK, POSE_CONN) 
        # Left hand connections
        self.mp_draw_util.draw_landmarks(frame, results.left_hand_landmarks, self.holistic_model.HAND_CONNECTIONS, HAND_LMARK, HAND_CONN) 
        # Right hand connections
        self.mp_draw_util.draw_landmarks(frame, results.right_hand_landmarks, self.holistic_model.HAND_CONNECTIONS, HAND_LMARK, HAND_CONN) 

    
    def record_gestures(self):
        sequence = []
        prediction_accuracy = 0
        threshold = 0.7
        self.load_model_weights()
        #Start recording process
        with self.holistic_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as h_model:
            while self.camera_capture.isOpened():

                # Read incoming camera frames
                ret, frame = self.camera_capture.read()

                #Detect and process gesture keypoints
                frame, results = self.keypoint_detection(frame, h_model)

                #Visualize keypoints and connections
                self.visualize_lmarks(frame, results)

                #Read landmark values
                lmark_values = self.read_lmark_values(results)
                sequence.append(lmark_values)

                cv2.rectangle(frame, (0, 550), (800, 600), (255, 0, 0), -1)
                
                #Predict gesture after 30 recorded frames
                if len(sequence) == 30:
                    res = self.model.predict(npy.expand_dims(sequence, axis=0))[0]
                    prediction_accuracy = res[npy.argmax(res)]
                    if prediction_accuracy < threshold:
                        sequence = []
                        continue
                    screen_text = "Gesture: " + self.gestures[npy.argmax(res)] + " Accuracy: " + str("%.2f" % prediction_accuracy)
                    
                    cv2.putText(frame, screen_text, (10, 585), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    cv2.imshow('Detecting gestures', frame)
                    cv2.waitKey(1000)
                    sequence = []
                    continue

                cv2.imshow('Detecting gestures', frame)

                # Quit capture gracefully by pressing ESC
                key = cv2.waitKey(10)
                if key == 27:
                    self.camera_capture.release()
                    cv2.destroyAllWindows()
                    return

if __name__ == '__main__':
    prediction = GesturePrediction()
    prediction.record_gestures()

