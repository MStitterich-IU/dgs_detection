import os

# Folder for recorded NumPy data
DATA_PATH = os.path.join('recording_data')

def setupStructure(gestures, no_videos, multicam=False):

    for gesture in gestures: 
        for video in range(1,no_videos+1):
            try:
                if multicam:
                    os.makedirs(os.path.join(DATA_PATH, gesture, str(video+no_videos)))
                os.makedirs(os.path.join(DATA_PATH, gesture, str(video)))
                
            except:
                pass

if __name__ == "__main__":
    gestures = ['sorry']
    setupStructure(gestures, 5)



