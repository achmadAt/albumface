import json
import uuid
import numpy as np
import cv2
import os
from deepface import DeepFace
import faiss

backends = ["retinaface", "dlib", "opencv"]
models = ["Facenet512"]

def initialize_folder(name: str):
    work_dir = os.getcwd()
    album_dir = work_dir + f"/{name}"
    if not os.path.exists(album_dir):
        os.makedirs(album_dir, exist_ok=True)
        print("Directory", album_dir, "created")

#ensure unique in similiar image
def ensure_unique(similiar_images):
    ids = set()
    result = []
    if len(similiar_images) != 0:
        for val in similiar_images:
            if val["id"] not in ids:
                ids.add(val["id"])
                result.append(val)
    
    return result


#Generate face image from photo
def generate_faces_image(path: str, album_dir: str):
    initialize_folder(album_dir)
    image_names = []
    extracted_face = DeepFace.extract_faces(img_path=path, enforce_detection=True, detector_backend=backends[0], align=True)
    for idx, face in enumerate(extracted_face):
        im = cv2.cvtColor(face['face'] * 255, cv2.COLOR_BGR2RGB)
        name = uuid.uuid4()
        cv2.imwrite(os.path.join(album_dir, f"{name}{idx}.jpg"), im)
        image_names.append(f"{name}{idx}.jpg")
    return image_names


#Get face embeddings from image
def generate_face_embeddings(path: str):
    embeddings = []
    embd = DeepFace.represent(img_path=path, detector_backend=backends[0], enforce_detection=True, model_name=models[0], align=False)
    for emb in embd:
        embeddings.append(emb['embedding'])
    return embeddings



def add_image_embd_to_image_db_json(image_data_json, emb, name):
    file_path = image_data_json
    existing_image_data = []
    image_id = str(uuid.uuid4())
    # emb = generate_face_embeddings(path=path)
    try:
        with open(file_path, "r") as infile:
            existing_image_data = json.load(infile)
    except FileNotFoundError:
        print("File not found. Creating a new one.")
    except json.decoder.JSONDecodeError:
        # Handle error
        pass
    new_data = {"id": image_id, "name": name, "embeddings": emb}
    #append data to json
    existing_image_data.append(new_data)
    with open(file_path, "w") as outfile:
        json.dump(existing_image_data, outfile, indent=4)
    return image_id

#Reading to image db json
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

#Reading to album db json
def read_album_db_json(album_data_json):
    file_path = album_data_json
    existing_album_data = []
    try:
        with open(file_path, "r") as infile:
            existing_album_data = json.load(infile)
    except FileNotFoundError:
        #This is only for read so pass
        pass
    except json.decoder.JSONDecodeError:
        # Handle error
        pass
    return existing_album_data


def calculate_image(person_embd, image_data):
    similiar_images = []
    person_embd = np.array(person_embd, dtype='f')
    person_embd = np.expand_dims(person_embd, axis=0)
    if len(image_data) != 0:
        d = 512
        k = len(image_data)
        index = faiss.IndexFlatIP(d)
        label = []
        #start adding image to faiss
        for idx, val in enumerate(image_data):
            for embd in val["embeddings"]:
                index.add(np.array([embd], dtype='f'))
                label.append(idx)
        
        dist, idx = index.search(person_embd, k)
        if len(idx[0]) != 0:
            for i, val in enumerate(idx[0]):
                if dist[0][i] >= 420.0:
                    similiar_images.append({"id": image_data[label[val]]["id"],"name": image_data[label[val]]["name"], "dist": float(dist[0][i])})
    return similiar_images

def calculate_album(person_embd, album_data):
    similiar_album = []
    person_embd = np.array(person_embd, dtype='f')
    person_embd = np.expand_dims(person_embd, axis=0)
    if len(album_data) != 0:
        d = 512
        k = len(album_data)
        index = faiss.IndexFlatIP(d)
        label = []
        #start adding image to faiss
        for idx, val in enumerate(album_data):
            index.add(np.array([val["embeddings"]], dtype='f'))
            label.append(idx)
        
        dist, idx = index.search(person_embd, k)
        if len(idx[0]) != 0:
            for i, val in enumerate(idx[0]):
                if dist[0][i] <= 410.0:
                    similiar_album.append(album_data[label[val]])
    return similiar_album

def calculate_album_selfie(person_embd, album_data):
    similiar_album = []
    person_embd = np.array(person_embd, dtype='f')
    person_embd = np.expand_dims(person_embd, axis=0)
    if len(album_data) != 0:
        d = 512
        k = len(album_data)
        index = faiss.IndexFlatIP(d)
        label = []
        #start adding image to faiss
        for idx, val in enumerate(album_data):
            index.add(np.array([val["embeddings"]], dtype='f'))
            label.append(idx)
        
        dist, idx = index.search(person_embd, k)
        if len(idx[0]) != 0:
            for i, val in enumerate(idx[0]):
                if dist[0][i] <= 410.0:
                    similiar_album.append({"album": album_data[label[val]], "dist": float(dist[0][i])})
    return similiar_album



def selfie_json_clean(album_found_data, selfie_data_json):
    clean_album_data = []
    if len(album_found_data) != 0:
        for idx, val in enumerate(album_found_data):
            album = {"id": val["album"]["id"], "similiar_images": val["album"]["similiar_images"]}
            clean_data = {"album": album, "dist": val["dist"]}
            clean_album_data.append(clean_data)
    with open(str(selfie_data_json) + "_clean.json", "w") as outfile:
        json.dump(clean_album_data, outfile, indent=4)

def album_json_clean(existing_album_data, album_data_json):
    clean_album_data = []
    if len(existing_album_data) != 0:
        for idx, val in enumerate(existing_album_data):
            clean_data = {"id": val["id"], "similiar_images": val["similiar_images"]}
            clean_album_data.append(clean_data)
    with open(str(album_data_json)+"_clean.json", "w") as outfile:
        json.dump(clean_album_data, outfile, indent=4)

def get_selfie_response(person_embd, album_data_json, selfie_data_json):
    album_found = [] 
    existing_album_data = read_album_db_json(album_data_json=album_data_json)
    album_found = calculate_album_selfie(person_embd=person_embd, album_data=existing_album_data)
    with open(selfie_data_json, "w") as outfile:
        json.dump(album_found, outfile, indent=4)
    selfie_json_clean(album_found_data=album_found, selfie_data_json=selfie_data_json)


def update_or_add_album(id_album, similiar_images, album_data_json, emb):
    existing_album_data = []
    existing_album_data = read_album_db_json(album_data_json=album_data_json)
    is_found = False
    arr = np.array(emb)
    embed = arr.tolist()
    if len(existing_album_data) != 0:
        for idx, val in enumerate(existing_album_data):
            if val["id"] == id_album:
                val["similiar_images"] = similiar_images
                is_found = True
    if is_found != True:
        new_data = {"id": id_album, "similiar_images": [similiar_images], "embeddings": embed}
        existing_album_data.append(new_data)
    with open(album_data_json, "w") as outfile:
        json.dump(existing_album_data, outfile, indent=4)
    album_json_clean(existing_album_data=existing_album_data, album_data_json=album_data_json)

def generate_album(path, album_data_json, image_data_json):
    face_names = generate_faces_image(path=path, album_dir="album")
    embd = generate_face_embeddings(path=path)
    image_id = add_image_embd_to_image_db_json(image_data_json=image_data_json, emb=embd, name=path)
    existing_album_data = read_album_db_json(album_data_json=album_data_json)
    image_data = read_image_db_json(image_data_json=image_data_json)
    for idx, emb in enumerate(embd):
        #add the origin as similiar
       similiar_images = calculate_image(person_embd=emb, image_data=image_data)
       similiar_images = ensure_unique(similiar_images=similiar_images)
       if len(similiar_images) == 0:
            similiar_images = [{"id": image_id, "name": path,"dist": float(500.0) }]
       album_found = calculate_album(person_embd=emb, album_data=existing_album_data)
       if len(album_found) != 0:
            id_album = album_found[0]["id"]
            similiar = album_found[0]["similiar_images"] = similiar_images
            update_or_add_album(id_album=id_album, album_data_json=album_data_json, similiar_images=similiar, emb=emb)
       else:
            update_or_add_album(id_album=face_names[idx], album_data_json=album_data_json, similiar_images=similiar_images, emb=emb)

def generate_face_detected_in_image(path):
    folder = "face"
    initialize_folder(folder)
    work_dir = os.getcwd()
    album_dir = work_dir + f"/{folder}/{path}"
    if not os.path.exists(album_dir):
        os.makedirs(album_dir, exist_ok=True)
        print("Directory", album_dir, "created")
    image_names = []
    extracted_face = DeepFace.extract_faces(img_path=path, enforce_detection=True, detector_backend=backends[0])
    for idx, face in enumerate(extracted_face):
        im = cv2.cvtColor(face['face'] * 255, cv2.COLOR_BGR2RGB)
        name = face["facial_area"]
        cv2.imwrite(os.path.join(album_dir, f"{str(name).strip()}.jpg"), im)
        image_names.append(f"face/{path}/{str(name).strip()}.jpg")
    return image_names

def read_face_db_json(json_path):
    file_path = json_path
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

def generate_face_json_data(image_path, json_path):
    existing_data = read_face_db_json(json_path=json_path)
    detected_face = generate_face_detected_in_image(path=image_path)
    embeddings_detected_face = generate_face_embeddings(path=image_path)
    for i, val in enumerate(embeddings_detected_face):
        existing_data.append({"name": detected_face[i], "embeddings": val})
    
    with open(json_path, "w") as outfile:
        json.dump(existing_data, outfile, indent=4)
        

def calculate_to_face(person_embd, json_path, json_output, image_path):
    #create image from input
    folder = "input"
    initialize_folder(folder)
    work_dir = os.getcwd()
    album_dir = work_dir + f"/{folder}"
    name = f"{folder}/{image_path}"
    extracted_face = DeepFace.extract_faces(img_path=image_path, enforce_detection=True, detector_backend=backends[0], align=True)
    for idx, face in enumerate(extracted_face):
        im = cv2.cvtColor(face['face'] * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(album_dir, f"{image_path}{idx}.jpg"), im)
    #calculate data
    data = read_face_db_json(json_path=json_path)
    person_embd = np.array(person_embd, dtype='f')
    person_embd = np.expand_dims(person_embd, axis=0)
    score_data = []
    if len(data) != 0:
        d = 512
        k = len(data)
        index = faiss.IndexFlatIP(d)
        label = []
        #start adding image to faiss
        for idx, val in enumerate(data):
            index.add(np.array([val["embeddings"]], dtype='f'))
            label.append(idx)
        
        dist, idx = index.search(person_embd, k)
        if len(idx[0]) != 0:
            for i, val in enumerate(idx[0]):
                score_data.append({"input": name,"data": data[label[val]]["name"], "score": float(dist[0][i]) + 40})
    with open(json_output, "w") as outfile:
        json.dump(score_data, outfile, indent=4)