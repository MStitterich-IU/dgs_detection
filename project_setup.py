import os

def setupStructure(gestures, no_videos):

    # Folder for recorded NumPy data
    DATA_PATH = os.path.join('recording_data')

    for gesture in gestures: 
        for video in range(1,no_videos+1):
            try:
                os.makedirs(os.path.join(DATA_PATH, gesture, str(video)))
                
            except:
                print("Passed")
                pass

if __name__ == "__main__":
    gestures = ['sorry']
    setupStructure(gestures, 5)



