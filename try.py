import json
import faiss

# Read the existing JSON data from the file
file_path = "data.json"
existing_data = []

try:
    with open(file_path, "r") as infile:
        existing_data = json.load(infile)
except FileNotFoundError:
    print("File not found. Creating a new one.")
except json.decoder.JSONDecodeError:
    # Handle error
    pass

if len(existing_data) != 0:
    for idx, val in enumerate(existing_data):
        print("step:", idx, "with val:", val)
        if existing_data[idx]['score'] <= 1:
            existing_data[idx]['similiar_images'].append(1)


new_object = {'id': 'test2', 'similiar_images': [], 'score': 0.021326946}
existing_data.append(new_object)

with open(file_path, "w") as outfile:
    json.dump(existing_data, outfile, indent=4)

index = faiss.IndexFlatIP(512)
