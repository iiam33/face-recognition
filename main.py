import os
import re
import cv2
from cv2.data import haarcascades
from pathlib import Path
from facedata_capture import FaceDataCapture
from model_training import ModelTraining


FACECASCADE = [os.path.join(haarcascades, f) for f in os.listdir(haarcascades) if
               re.match("haarcascade_frontalface_default.xml", f)][0]

DETECTOR = cv2.CascadeClassifier(FACECASCADE)


class Main:
    def main():
        working_path = Path.cwd()
        train_data_path = os.path.join(working_path, 'data', 'train')
        no_of_users = input('Enter number of face to be detected: ')

        counter = 0
        names = []

        while counter < int(no_of_users):
            name = input(f'Enter User {counter+1} Name: ')
            names.append(name)
            img_dir = os.path.join(train_data_path, name)
            FaceDataCapture.capture_face_data(
                user_data_path=img_dir, detector=DETECTOR)
            counter += 1

        ModelTraining.train_model(
            train_data_path=train_data_path, detector=DETECTOR)

    if __name__ == '__main__':
        main()
