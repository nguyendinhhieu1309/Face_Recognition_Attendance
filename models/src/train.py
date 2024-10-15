import os
import cv2
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def train_new_model():
    faces = []
    Id = []
    IDUnique = []

    processed_data_path = 'F:\SUMMER24\CPV301\Face_Recognition_Attendance_System\processed_data'
    if not os.path.exists(processed_data_path):
        print(f"Processed data directory does not exist: {processed_data_path}")
        return

    for subdir in os.listdir(processed_data_path):
        subdir_path = os.path.join(processed_data_path, subdir)
        if os.path.isdir(subdir_path):
            IDUnique.append(subdir)

    for id in IDUnique:
        id_path = os.path.join(processed_data_path, id)
        for facedir in os.listdir(id_path):
            face_path = os.path.join(id_path, facedir)
            face = cv2.imread(face_path)
            if face is not None:
                faces.append(face)
                Id.append(id)
            else:
                print(f"Failed to read image: {face_path}")

    if not faces:
        print("No faces found for training.")
        return

    embedder = FaceNet()
    embeddings = embedder.embeddings(faces)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, Id, test_size=0.1, random_state=42)

    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(X_train)
    testX = in_encoder.transform(X_test)

    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    trainy = out_encoder.transform(y_train)
    testy = out_encoder.transform(y_test)

    print(f"Encoded labels: {out_encoder.classes_}")

    model = SVC(kernel='linear')
    model.fit(trainX, trainy)

    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)

    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)

    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    models_path = 'F:\SUMMER24\CPV301\Face_Recognition_Attendance_System/models/weights'
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        print(f"Created directory: {models_path}")

    with open(os.path.join(models_path, 'svm_saved.sav'), 'wb') as file:
        pickle.dump(model, file)
    with open(os.path.join(models_path, 'out_encoder.pkl'), 'wb') as file:
        pickle.dump(out_encoder, file)
    print("Model and encoder saved successfully.")