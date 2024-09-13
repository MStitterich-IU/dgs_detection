import os


def setup_structure(data_path, gestures, video_count, multicam=False):
    
    for gesture in gestures: 
        max_dir = 0
        try:
            existing_dir = os.listdir(os.path.join(data_path, gesture))
            max_dir = max(list(map(int, existing_dir)))
        except:
            pass

        for video in range(max_dir + 1, max_dir + video_count+1):
            try:
                if multicam:
                    os.makedirs(os.path.join(data_path, gesture, str(video+video_count)))
                os.makedirs(os.path.join(data_path, gesture, str(video)))
                
            except:
                pass

if __name__ == "__main__":
    gestures = ['danke']
    setup_structure(gestures, 5)



