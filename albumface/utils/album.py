import json
import uuid
import cv2
import os
from deepface import DeepFace


backends = ["retinaface", "dlib", "opencv"]
models = ["Facenet512"]

def initialize_folder(name: str):
    work_dir = os.getcwd()
    album_dir = work_dir + f"/{name}"
    if not os.path.exists(album_dir):
        os.makedirs(album_dir, exist_ok=True)
        print("Directory", album_dir, "created")

def generate_faces_image(path: str, album_dir: str):
    initialize_folder(album_dir)
    extracted_face = DeepFace.extract_faces(img_path=path, enforce_detection=True, detector_backend=backends[0], align=True)
    for idx, face in enumerate(extracted_face):
        im = cv2.cvtColor(face['face'] * 255, cv2.COLOR_BGR2RGB)
        name = uuid.uuid4()
        cv2.imwrite(os.path.join(album_dir, f"{name}{idx}.jpg"), im)

def generate_face_embeddings(path: str):
    embeddings = []
    embd = DeepFace.represent(img_path=path, detector_backend=backends[0], enforce_detection=True, model_name=models[0])
    for emb in embd:
        embeddings.append(emb['embedding'])
    return embeddings

def generate_album(image_path, album_json_path, imgs_vector_json):
    return "Hello"

def calculate_albums(path):
    return "World"

def calculate_image(path):
    return "Hello"