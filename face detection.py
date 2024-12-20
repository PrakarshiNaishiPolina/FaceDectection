
import cv2 as cv
import mediapipe as mp

face_detection=mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


cap=cv.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    # flip the frame horizontally 
    frame=cv.flip(frame,1)

    # convert the image to RGB
    rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    # process the frame 

    results=face_detection.process(rgb_frame)

    # if face is found, draw landmarks

    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame,detection)

    cv.imshow("Face Detection",frame)

#break

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

# release video capture and close any opencv windows

cap.release()
cv.destroyAllWindows()
