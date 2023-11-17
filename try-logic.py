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

def generate_image_db_json(path, image_data_json):
    file_path = image_data_json
    existing_image_data = []
    emb = generate_face_embeddings(path=path)
    try:
        with open(file_path, "r") as infile:
            existing_image_data = json.load(infile)
    except FileNotFoundError:
        print("File not found. Creating a new one.")
    except json.decoder.JSONDecodeError:
        # Handle error
        pass
    new_data = {"id": str(uuid.uuid4()), "name": path, "embeddings": emb}
    #append data to json
    existing_image_data.append(new_data)
    with open(file_path, "w") as outfile:
        json.dump(existing_image_data, outfile, indent=4)

def read_image_db_json(image_data_json):
    file_path = image_data_json
    existing_data = []
    try:
        with open(file_path, "r") as infile:
            existing_data = json.load(infile)
    except FileNotFoundError:
        #This is only for read so pass
        pass
    except json.decoder.JSONDecodeError:
        # Handle error
        pass
    return existing_data

# generate_faces_image(path="img1.jpg", album_dir="album")

# emb = generate_face_embeddings(path="img1.jpg")

file_test = "data_image.json"
generate_image_db_json("img1.jpg", file_test)

data = read_image_db_json(file_test)
print(data)