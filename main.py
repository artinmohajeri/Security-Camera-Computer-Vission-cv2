from playsound import playsound
import cv2, time, datetime
import numpy as np


cap = cv2.VideoCapture(0)
detection = False
detection_stopped_time = None
timer_starter = False
SECONDS_TO_RECORD_AFTER_DETECTION = 7

# record a video
frame_size = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

while True:
    _, frame = cap.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    bodies = body_cascade.detectMultiScale(grayscale, 1.3, 5)
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_starter = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20.0, frame_size)
            print("Start Recording...")
            playsound('./alert_sound.mp3')
    elif detection:
        if timer_starter:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_starter = False
                out.release()
                print("Stop Recording")
        else:
            timer_starter = True
            detection_stopped_time = time.time()
    if detection:
        out.write(frame)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
