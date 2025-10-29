from detection import FaceDetector
from extraction import FaceEmbedder
from matcher import FaceMatcher
from utils import loadReference
from database import save_face
import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    dataset_dir = "resources/references"
    output_dir = "resources/output"
    os.makedirs(output_dir, exist_ok=True)
    input_path = "resources/input/harrym.jpg"

    # Inisialisasi model InsightFace
    detector = FaceDetector(ctx_id=-1)
    embedder = FaceEmbedder(detector.app)
    matcher = FaceMatcher(threshold=0.7)

    # Proses gambar referensi
    references = loadReference(dataset_dir)
    print(f"[DEBUG] Total reference images loaded: {len(references)}")
    for name, _ in references:
        print(f"[DEBUG] Loaded: {name}")
    reference_embeddings = {}

    for name, img in references:
        print(f"[DEBUG] Checking {name}...")
        faces = detector.detectFaces(img)
        print(f"[DEBUG] {name}: {len(faces)} face(s) detected")
        if not faces:
            print(f"[WARN] No face found in {name}")
            continue
        emb = embedder.getEmbedding(faces[0])
        reference_embeddings[name] = emb
        print(f"[INFO] Processed reference: {name}, saved {len(faces)} face(s)")

        # Get user name
        user_name = name.split("_")[0]
        save_face(user_name, os.path.join(dataset_dir, name), emb)

    # Proses gambar input
    input_img = cv2.imread(input_path)
    input_faces = detector.detectFaces(input_img)
    input_emb = embedder.getEmbedding(input_faces[0])

    # Cari match terbaik
    best_name, distance = matcher.getMatching(input_emb, reference_embeddings, verbose=True)
    print(f"Best match: {best_name}, Distance: {distance:.2f}")

    if distance < matcher.threshold:
        print("SAME PERSON")
    else:
        print("DIFFERENT PERSON")

if __name__ == "__main__":
    main()