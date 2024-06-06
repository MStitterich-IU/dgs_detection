import cv2
import mediapipe as mp

holisticModel = mp.solutions.holistic
mpDrawUtil = mp.solutions.drawing_utils

def keypoint_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False                  
    results = model.process(frame)                 
    frame.flags.writeable = True                    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, results

cameraCapture = cv2.VideoCapture(1)
with holisticModel.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holisticModel:
    while cameraCapture.isOpened():
        # Read incoming camera frames
        ret, frame = cameraCapture.read()

        #Detect and process gesture keypoints
        frame, results = keypoint_detection(frame, holisticModel)

        # Show camera feed to user
        cv2.imshow('Detect gesture keypoints', frame)

        # Quit capture gracefully via key input
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
cameraCapture.release()
cv2.destroyAllWindows()