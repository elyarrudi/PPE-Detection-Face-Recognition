import os

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# ======================================================
# Paths
# ======================================================
DOCKER_PROJECT_DIR = "/app/FaceRecognition"
LOCAL_PROJECT_DIR = "FaceRecognition"

if os.path.isdir(DOCKER_PROJECT_DIR):
    PROJECT_DIR = DOCKER_PROJECT_DIR
elif os.path.isdir(LOCAL_PROJECT_DIR):
    PROJECT_DIR = LOCAL_PROJECT_DIR
else:
    PROJECT_DIR = "."

REGISTER_DIR = os.path.join(PROJECT_DIR, "register_images")
DATABASE_DIR = os.path.join(PROJECT_DIR, "database")

if not os.path.isdir(REGISTER_DIR):
    raise FileNotFoundError(f"Registration image directory not found: {REGISTER_DIR}")

os.makedirs(DATABASE_DIR, exist_ok=True)

# ======================================================
# Load InsightFace model
# ======================================================
CUDA_PROVIDER = "CUDAExecutionProvider"
if CUDA_PROVIDER in ort.get_available_providers():
    providers = [CUDA_PROVIDER, "CPUExecutionProvider"]
    context_id = 0
else:
    providers = ["CPUExecutionProvider"]
    context_id = -1

app = FaceAnalysis(
    name="buffalo_l",
    providers=providers,
)
app.prepare(
    ctx_id=context_id,
    det_size=(640, 640),
)
print(f"InsightFace loaded successfully ({'GPU' if context_id == 0 else 'CPU'})")

# ======================================================
# Process each person
# ======================================================
persons = sorted(os.listdir(REGISTER_DIR))
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for person_name in persons:
    person_path = os.path.join(REGISTER_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    print("=" * 50)
    print(f"Person: {person_name}")
    embeddings = []

    images = sorted(
        image_name
        for image_name in os.listdir(person_path)
        if os.path.splitext(image_name)[1].lower() in valid_extensions
    )

    # ==================================================
    # Process images for the person
    # ==================================================
    for image_name in images:
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Cannot read: {image_name}")
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"No face found: {image_name}")
            continue

        if len(faces) != 1:
            print(f"Invalid image (multiple faces): {image_name}")
            continue

        embedding = faces[0].embedding
        embeddings.append(embedding)
        print(f"{image_name} -> face found")

    # ==================================================
    # Create the final person embedding
    # ==================================================
    if len(embeddings) == 0:
        print(f"No valid images for: {person_name}")
        continue

    mean_embedding = np.mean(np.array(embeddings), axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    # ==================================================
    # Save the embedding to the database
    # ==================================================
    save_folder = os.path.join(DATABASE_DIR, person_name)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, "embedding.npy")
    np.save(save_path, mean_embedding)
    print(f"Embedding saved: {save_path}")

print("=" * 50)
print("Database generation finished.")
