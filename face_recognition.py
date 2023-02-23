import os
import re
import cv2
from cv2.data import haarcascades
from pathlib import Path


class FaceRecognizor:
    def recognize_face(names):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        working_path = Path.cwd()
        model_path = os.path.join(working_path, "model", "trainning_data.yml")
        recognizer.read(model_path)

        face_casade = [os.path.join(haarcascades, f) for f in os.listdir(haarcascades) if
                       re.match("haarcascade_frontalface_default.xml", f)][0]

        detector = cv2.CascadeClassifier(face_casade)

        video = cv2.VideoCapture(0)
        
        names = ['alvin', 'jas']

        while True:
            rval, frame = video.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(20, 20)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(
                        round(number=confidence, ndigits=2))
                else:
                    id = "Unknown"
                    confidence = "  {0}%".format(
                        round(number=confidence-100, ndigits=2))

                cv2.putText(
                    img=frame,
                    text=str(id),
                    org=(x+5, y-5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                    bottomLeftOrigin=False,
                )
                cv2.putText(
                    img=frame,
                    text=str(confidence),
                    org=(x+5, y+h-5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    bottomLeftOrigin=False,
                )

            cv2.imshow('camera', frame)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        video.release()
        cv2.destroyWindow("video")

    if __name__ == "__main__":
        working_path = Path.cwd()
        train_data_path = os.path.join(working_path, 'data', 'train')
        user_path = [f.path for f in os.scandir(train_data_path) if f.is_dir()]
        users = [user.split('/')[-1] for user in user_path]

        recognize_face(names=users)
