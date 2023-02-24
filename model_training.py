import os
import re
import cv2
import numpy as np
from cv2.data import haarcascades
from pathlib import Path
from PIL import Image


class ModelTraining:
    def get_face_data_id(path, detector):
        users = [f.path for f in os.scandir(path) if f.is_dir()]

        face_samples = []
        face_ids = []

        for idx, user in enumerate(users):
            image_paths = [os.path.join(user, f)
                           for user, dirs, files in os.walk(user) for f in files]

            for image_path in image_paths:
                PIL_img = Image.open(image_path).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                id = idx
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    face_ids.append(id)

        return face_samples, face_ids

    def train_model(detector, train_data_path):

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        print("\n [INFO] Training faces into model...")
        faces, face_ids = ModelTraining.get_face_data_id(
            path=train_data_path, detector=detector)

        recognizer.train(faces, np.array(face_ids))

        working_path = Path.cwd()
        model_path = os.path.join(working_path, "model")
        os.makedirs(model_path, exist_ok=True)
        model_save_path = os.path.join(model_path, "trainning_data.yml")
        recognizer.write(model_save_path)

        print("\n [INFO] {0} faces trained.".format(
            len(np.unique(face_ids))))
