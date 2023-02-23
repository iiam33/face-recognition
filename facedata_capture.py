import os
import re
import cv2
from cv2.data import haarcascades

TRAIN_DATA_COUNT = 200


class FaceDataCapture:
    def capture_face_data(user_data_path, detector):
        img_count = TRAIN_DATA_COUNT
        os.makedirs(user_data_path, exist_ok=True)

        cv2.namedWindow("Capturing...")
        video = cv2.VideoCapture(0)

        fname = len(os.listdir(user_data_path))
        count = 0

        print("\n [INFO] Capturing face data...")
        while count < img_count:
            rval, frame = video.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(user_data_path, str(
                    fname) + ".jpg"), gray[y:y+h, x:x+w])

                fname += 1
                count += 1
                cv2.putText(
                    img=frame,
                    text=f'Scanning: {((fname / img_count) * 100):.2f}%',
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                    bottomLeftOrigin=False,
                )

                cv2.imshow('Capturing...', frame)

                key = cv2.waitKey(20)

                if key == 27:
                    break
                elif count >= img_count:
                    break

        video.release()
        cv2.destroyWindow("video")

    if __name__ == '__main__':
        capture_face_data()
