from record_gestures import DataRecorder
from model_training import Model
import cv2
import numpy as npy

class GesturePrediction(DataRecorder, Model):

    def __init__(self):
        DataRecorder.__init__(self)
        Model.__init__(self, data_folder='testing_data')
    
    def record_gestures(self):
        predictions, sequence = [], []
        self.model.load_weights('good.keras')
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
                
                #Predict gesture after 30 recorded frames
                if len(sequence) == 30:
                    res = self.model.predict(npy.expand_dims(sequence, axis=0))[0]
                    predictions.append(npy.argmax(res))
                    predictions = predictions [-5:]
                    #Make sure that prediction is not just a fluke
                    if npy.unique(predictions[-2:])[0] == npy.argmax(res):
                        print(self.gestures[npy.argmax(res)])
                    sequence = []
                
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

