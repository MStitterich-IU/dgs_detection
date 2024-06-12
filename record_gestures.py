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
    poseValues = npy.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else npy.zeros(33*4)
    faceValues = npy.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else npy.zeros(468*3)
    lhValues = npy.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else npy.zeros(21*3)
    rhValues = npy.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else npy.zeros(21*3)
    return npy.concatenate([poseValues, faceValues, lhValues, rhValues])

#Set up camera for recording
cameraCapture = cv2.VideoCapture(1)
cameraCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cameraCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

def record_gestures(recordingGestures, videoCount=5, framesPerVideo=30):

    #Create project recording folders
    pSetup.setupStructure(recordingGestures, videoCount)

    #Start recording process
    with holisticModel.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hModel:
        
        for gesture in recordingGestures:
            existingDir = os.listdir(os.path.join(pSetup.DATA_PATH, gesture))
            maxDir = max(list(map(int, existingDir)))            
            for video in range(maxDir-videoCount+1, maxDir+1):
                for frameNr in range(1, framesPerVideo+1):
                    # Read incoming camera frames
                    ret, frame = cameraCapture.read()

                    #Detect and process gesture keypoints
                    frame, results = keypoint_detection(frame, hModel)

                    #Visualize keypoints and connections
                    visualize_lmarks(frame, results)

                    #Wait for key input before starting data collection for each video sequence
                    if frameNr == 1: 
                        cv2.putText(frame, 'Press key to start collecting data', (30,30),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, 'for gesture {} video number {}'.format(gesture, video),
                                    (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Detect gesture keypoints', frame)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(frame, 'Collecting data for gesture', (30,30),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, '{} video number {}'.format(gesture, video),
                                    (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Detect gesture keypoints', frame)

                    #Read landmark values and save to file
                    lmarkValues = read_lmark_values(results)
                    filePath = os.path.join(pSetup.DATA_PATH, gesture, str(video), str(frameNr))
                    npy.save(filePath, lmarkValues)

                    # Quit capture gracefully by pressing ESC
                    key = cv2.waitKey(10)
                    if key == 27:
                        return

        cameraCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recordingGestures = []
    gesture = input("Welche Geste soll aufgenommen werden?\n")
    count = int(input("Wie viele Videos pro Geste?\n"))
    recordingGestures.append(gesture)
    record_gestures(recordingGestures, count)
    print('Recording finished')
