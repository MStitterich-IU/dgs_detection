import os

# Folder for recorded NumPy data
DATA_PATH = os.path.join('recording_data')

def setupStructure(gestures, videoCount, multicam=False):
    
    for gesture in gestures: 
        existingMaxDir = 0
        try:
            existingMaxDir = int(max(os.listdir(os.path.join(DATA_PATH, gesture))))            
        except:
            pass

        for video in range(existingMaxDir + 1, existingMaxDir + videoCount+1):
            try:
                if multicam:
                    os.makedirs(os.path.join(DATA_PATH, gesture, str(video+videoCount)))
                os.makedirs(os.path.join(DATA_PATH, gesture, str(video)))
                
            except:
                pass

if __name__ == "__main__":
    gestures = ['danke']
    setupStructure(gestures, 5)



