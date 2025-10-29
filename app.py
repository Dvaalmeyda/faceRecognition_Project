from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import psycopg2
import cv2
import numpy as np
import uuid
from detection import FaceDetector
from extraction import FaceEmbedder
from matcher import FaceMatcher
from database import load_all_faces
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="Face Recognition API")

# Inisialisasi model InsightFace
detector = FaceDetector(ctx_id=-1)
embedder = FaceEmbedder(detector.app)
matcher = FaceMatcher(threshold=0.8)

# Connect Postgre
conn = psycopg2.connect(
    host="localhost",
    database="facerec_db",
    user="postgres",
    password="1234"
)
cursor = conn.cursor()

# Cek Tabel
def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL UNIQUE
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_images (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        image_path TEXT NOT NULL,
        embedding BYTEA NOT NULL
    )
    """)
    conn.commit()
    print("[INFO] Database tables checked/created")

init_db()

# Pydantic for register
class FaceRegister(BaseModel):
    name: str

# Helper Function : Load Embeddings
def load_faces():
    cursor.execute("""
        SELECT u.id, u.name, f.embedding
        FROM users u
        JOIN face_images f ON u.id = f.user_id
    """)
    rows = cursor.fetchall()
    result = {}
    for user_id, name, emb_bytes in rows:
        emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
        if name not in result:
            result[name] = []
        result[name].append((user_id, emb_array))
    return result

@app.get("/api/face")
def get_faces():
    cursor.execute("""
        SELECT u.id, u.name, f.id as face_id, f.image_path
        FROM users u
        JOIN face_images f ON u.id = f.user_id
    """)
    rows = cursor.fetchall()
    faces = [{"user_id": row[0], "name": row[1], "face_id": row[2], "image_path": row[3]} for row in rows]
    return {"faces": faces}

@app.post("/api/face/register")
async def register_face(file: UploadFile = File(...), name: str = ""):
    if name == "":
        raise HTTPException(status_code=400, detail="Name is required")

    # Cek user
    cursor.execute("SELECT id FROM users WHERE name=%s", (name,))
    user = cursor.fetchone()
    if user is None:
        cursor.execute("INSERT INTO users (name) VALUES (%s) RETURNING id", (name,))
        user_id = cursor.fetchone()[0]
        conn.commit()
    else:
        user_id = user[0]

    # Read img
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = detector.detectFaces(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected")

    embedding = embedder.getEmbedding(faces[0])

    # save
    image_path = f"resources/references/{uuid.uuid4()}.jpg"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cv2.imwrite(image_path, img)

    # save embedding
    cursor.execute(
        "INSERT INTO face_images (user_id, image_path, embedding) VALUES (%s, %s, %s) RETURNING id",
        (user_id, image_path, embedding.tobytes())
    )
    face_id = cursor.fetchone()[0]
    conn.commit()

    return {"user_id": user_id, "face_id": face_id, "name": name, "message": "Face registered successfully"}

@app.post("/api/face/recognize")
async def face_recognition(file: UploadFile = File(...)):
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = detector.detectFaces(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected")

    input_emb = embedder.getEmbedding(faces[0])
    reference_embeddings = load_faces()

    best_match_name = None
    best_distance = float("inf")
    best_user_id = None

    for name, emb_list in reference_embeddings.items():
        for user_id, emb_array in emb_list:
            dist = float(np.linalg.norm(input_emb - emb_array))  # float murni
            if dist < best_distance:
                best_distance = dist
                best_match_name = name
                best_user_id = user_id

    if best_distance < matcher.threshold:
        status = "Same Person"
    else:
        status = "Different Person"
        best_match_name = None
        best_user_id = None

    return {
        "user_id": int(best_user_id) if best_user_id is not None else None,
        "name": best_match_name,
        "distance": float(best_distance),
        "status": status
    }

@app.delete("/api/face/{face_id}")
def delete_face(face_id: int):
    cursor.execute("SELECT image_path FROM face_images WHERE id=%s", (face_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Face ID not found")

    image_path = row[0]
    if os.path.exists(image_path):
        os.remove(image_path)

    cursor.execute("DELETE FROM face_images WHERE id=%s", (face_id,))
    conn.commit()
    return {"message": f"Face {face_id} deleted successfully"}