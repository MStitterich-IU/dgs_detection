import cv2

cameraCapture = cv2.VideoCapture(1)


while cameraCapture.isOpened():
    # Read incoming camera frames
    ret, frame = cameraCapture.read()

    # Show camera feed to user
    cv2.imshow('OpenCV Feed', cv2.flip(frame, 180))

    # Quit capture gracefully via key input
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
cameraCapture.release()
cv2.destroyAllWindows()