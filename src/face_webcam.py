import cv2


vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/Users/talhayilmaz/Desktop/OpenCv/Opp/haarCascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/talhayilmaz/Desktop/OpenCv/Opp/haarCascade/haarcascade_eye.xml")

while True:
    _,frame = vid.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,7)

    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),7)

    roi_frame = frame[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray,1.3,15)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(233,233,75),3)

    cv2.imshow("video", frame)
    if cv2.waitKey(10) & 0xff == ord("q"):
        break

vid.release
cv2.destroyAllWindows()
