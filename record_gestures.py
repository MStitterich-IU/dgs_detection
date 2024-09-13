from contextlib import nullcontext
import cv2
import mediapipe as mp
import numpy as npy
import os
import project_setup
import tkinter as tk
from tkinter import filedialog, simpledialog

class DataRecorder():
    
    def __init__(self, data_path=None, multicam=False):

        self.data_path = data_path
        self.multicam = multicam

        #Mediapipe holistic model and utilites for keypoint detection and visualization
        self.holistic_model = mp.solutions.holistic
        self.mp_draw_util = mp.solutions.drawing_utils

        #Set up camera for recording
        self.camera_capture = cv2.VideoCapture(1)
        self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        #Multi-camera setup
        if self.multicam:
            self.holistic_model2 = mp.solutions.holistic
            self.mp_draw_util2 = mp.solutions.drawing_utils

            self.camera_capture2 = cv2.VideoCapture(0)
            self.camera_capture2.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.camera_capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    def keypoint_detection(self, frame, model):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = model.process(frame)                 
        frame.flags.writeable = True                    
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame, results

    def visualize_lmarks(self, frame, results, cam_nr=1):
        if cam_nr == 2:
            #Drawing specs for changing the  presentation style
            POSE_LMARK = self.mp_draw_util2.DrawingSpec(color=(168,52,50), thickness=2, circle_radius=3)
            POSE_CONN = self.mp_draw_util2.DrawingSpec(color=(50,50,168), thickness=2, circle_radius=2)
            HAND_LMARK = self.mp_draw_util2.DrawingSpec(color=(134,109,29), thickness=2, circle_radius=3)
            HAND_CONN = self.mp_draw_util2.DrawingSpec(color=(161,161,160), thickness=2, circle_radius=2)

            # Visualize face contours
            #mpDrawUtil.draw_landmarks(frame, results.face_landmarks, holisticModel.FACEMESH_CONTOURS)
            
            # Posture landmarks
            self.mp_draw_util2.draw_landmarks(frame, results.pose_landmarks, self.holistic_model2.POSE_CONNECTIONS, POSE_LMARK, POSE_CONN) 
            # Left hand connections
            self.mp_draw_util2.draw_landmarks(frame, results.left_hand_landmarks, self.holistic_model2.HAND_CONNECTIONS, HAND_LMARK, HAND_CONN) 
            # Right hand connections
            self.mp_draw_util2.draw_landmarks(frame, results.right_hand_landmarks, self.holistic_model2.HAND_CONNECTIONS, HAND_LMARK, HAND_CONN) 

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

    def read_lmark_values(self, results):
        pose_values = npy.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else npy.zeros(33*4)
        face_values = npy.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else npy.zeros(468*3)
        lh_values = npy.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else npy.zeros(21*3)
        rh_values = npy.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else npy.zeros(21*3)
        return npy.concatenate([pose_values, face_values, lh_values, rh_values])
    
    def record_gestures(self, recording_gestures, video_count=5, frames_per_video=30):
        #Create project recording folders
        project_setup.setup_structure(self.data_path, recording_gestures, video_count, self.multicam)

        #Start recording process
        with self.holistic_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as h_model:
            with self.holistic_model2.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) if self.multicam else nullcontext() as h_model2:
                for gesture in recording_gestures:

                    existing_dir = os.listdir(os.path.join(self.data_path, gesture))
                    max_dir = max(list(map(int, existing_dir)))
                    if self.multicam:
                        range_start = (max_dir-video_count*2)+1
                        range_stop = max_dir-video_count+ 1
                    else:
                        range_start = max_dir-video_count+1
                        range_stop = max_dir+1

                    for video in range(range_start, range_stop):
                        for frame_nr in range(1, frames_per_video+1):
                            # Read incoming camera frames
                            ret, frame = self.camera_capture.read()
                            if self.multicam:
                                ret2, frame2 = self.camera_capture2.read()

                            #Detect and process gesture keypoints
                            frame, results = self.keypoint_detection(frame, h_model)
                            if self.multicam:
                                frame2, results2 = self.keypoint_detection(frame2, h_model2)

                            #Visualize keypoints and connections
                            self.visualize_lmarks(frame, results)
                            if self.multicam:
                                self.visualize_lmarks(frame2, results2, cam_nr=2)

                            #Wait for key input before starting data collection for each video sequence
                            if frame_nr == 1: 
                                cv2.putText(frame, 'Press key to start collecting data', (30,30),
                                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                cv2.putText(frame, 'for gesture "{}"; Video number {}'.format(gesture, video),
                                            (30, 65), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                cv2.imshow('Camera 1', frame)
                                if self.multicam:
                                    cv2.imshow('Camera 2', frame2)
                                cv2.waitKey(2000)
                            else:
                                cv2.putText(frame, 'Collecting data for gesture', (30,30),
                                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                cv2.putText(frame, '"{}"; Video number {}'.format(gesture, video),
                                            (30, 65), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                cv2.imshow('Camera 1', frame)
                                if self.multicam:
                                    cv2.imshow('Camera 2', frame2)

                            #Read landmark values and save to file
                            lmark_values = self.read_lmark_values(results)
                            file_path = os.path.join(self.data_path, gesture, str(video), str(frame_nr))
                            npy.save(file_path, lmark_values)
                            if self.multicam:
                                lmark_values2 = self.read_lmark_values(results2)
                                file_path = os.path.join(self.data_path, gesture, str(video+video_count), str(frame_nr))
                                npy.save(file_path, lmark_values2)

                            # Quit capture gracefully by pressing ESC
                            key = cv2.waitKey(10)
                            if key == 27:
                                return

            self.camera_capture.release()
            if self.multicam:
                self.camera_capture2.release()
            cv2.destroyAllWindows()

def get_user_input():
    recording_gestures = []
    root = tk.Tk()
    root.withdraw()
    gesture = simpledialog.askstring("Gesture", "What gesture do you want to record?")
    count = simpledialog.askinteger("Video Count", "How many training videos do you want to record?")
    recording_gestures.append(gesture)
    file_path = filedialog.askdirectory(title="Please select the data folder")
    return recording_gestures, count, file_path

if __name__ == '__main__':
    gestures, count, file_path = get_user_input()
    recorder = DataRecorder(data_path=file_path, multicam=False)
    recorder.record_gestures(gestures, count)
    print('Recording finished')
 