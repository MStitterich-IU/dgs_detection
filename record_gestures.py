import project_setup as pSetup
import cv2
import mediapipe as mp
import numpy as npy
import os

holisticModel = mp.solutions.holistic
mpDrawUtil = mp.solutions.drawing_utils

def keypoint_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False                  
    results = model.process(frame)                 
    frame.flags.writeable = True                    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, results

def visualize_lmarks(frame, results):
    #Drawing specs for changing the  presentation style
    POSE_LMARK = mpDrawUtil.DrawingSpec(color=(168,52,50), thickness=2, circle_radius=3)
    POSE_CONN = mpDrawUtil.DrawingSpec(color=(50,50,168), thickness=2, circle_radius=2)
    HAND_LMARK = mpDrawUtil.DrawingSpec(color=(134,109,29), thickness=2, circle_radius=3)
    HAND_CONN = mpDrawUtil.DrawingSpec(color=(161,161,160), thickness=2, circle_radius=2)

    # Visualize face contours
    #mpDrawUtil.draw_landmarks(frame, results.face_landmarks, holisticModel.FACEMESH_CONTOURS)
    
    # Posture landmarks
    mpDrawUtil.draw_landmarks(frame, results.pose_landmarks, holisticModel.POSE_CONNECTIONS, POSE_LMARK, POSE_CONN) 
    # Left hand connections
    mpDrawUtil.draw_landmarks(frame, results.left_hand_landmarks, holisticModel.HAND_CONNECTIONS, HAND_LMARK, HAND_CONN) 
    # Right hand connections
    mpDrawUtil.draw_landmarks(frame, results.right_hand_landmarks, holisticModel.HAND_CONNECTIONS, HAND_LMARK, HAND_CONN) 

def read_lmark_values(results):
    poseValues = npy.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    faceValues = npy.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    lhValues = npy.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    rhValues = npy.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    return npy.concatenate([poseValues, faceValues, lhValues, rhValues])

cameraCapture = cv2.VideoCapture(1)

with holisticModel.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hModel:
    while cameraCapture.isOpened():
        # Read incoming camera frames
        ret, frame = cameraCapture.read()

        #Detect and process gesture keypoints
        frame, results = keypoint_detection(frame, hModel)

        #Visualize keypoints and connections
        visualize_lmarks(frame, results)

        # Show camera feed to user
        cv2.imshow('Detect gesture keypoints', frame)

        #Read landmark values and save to file
        lmarkValues = read_lmark_values(results)
        filePath = os.path.join(pSetup.DATA_PATH, "test")
        npy.save(filePath, lmarkValues)

        # Quit capture gracefully via key input
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
cameraCapture.release()
cv2.destroyAllWindows()