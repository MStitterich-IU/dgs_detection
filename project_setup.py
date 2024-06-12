import os

# Folder for recorded NumPy data
DATA_PATH = os.path.join('recording_data')

def setupStructure(gestures, videoCount, multicam=False):
    
    for gesture in gestures: 
        maxDir = 0
        try:
            existingDir = os.listdir(os.path.join(DATA_PATH, gesture))
            maxDir = max(list(map(int, existingDir)))
        except:
            pass

        for video in range(maxDir + 1, maxDir + videoCount+1):
            try:
                if multicam:
                    os.makedirs(os.path.join(DATA_PATH, gesture, str(video+videoCount)))
                os.makedirs(os.path.join(DATA_PATH, gesture, str(video)))
                
            except:
                pass

if __name__ == "__main__":
    gestures = ['danke']
    setupStructure(gestures, 5)



