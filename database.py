import psycopg2
import numpy as np

# Connect PostgreSQL
def connect_db():
    conn = psycopg2.connect(
        host="localhost",
        database="facerec_db",
        user="postgres",
        password="1234",
        port="5432"
    )
    return conn

def add_user(name):
    """Tambah user baru ke tabel users kalau belum ada"""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE name = %s", (name,))
    user = cur.fetchone()

    if user:
        user_id = user[0]
    else:
        cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id", (name,))
        user_id = cur.fetchone()[0]
        print(f"[INFO] Added new user: {name}")

    conn.commit()
    cur.close()
    conn.close()
    return user_id

def save_face(user_name, image_path, embedding):
    """Simpan embedding wajah ke tabel face_images"""
    conn = connect_db()
    cur = conn.cursor()

    user_id = add_user(user_name)
    emb_bytes = embedding.astype(np.float32).tobytes()

    cur.execute(
        "INSERT INTO face_images (user_id, image_path, embedding) VALUES (%s, %s, %s)",
        (user_id, image_path, emb_bytes)
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"[INFO] Saved embedding for {user_name} -> {image_path}")

def load_all_faces():
    """Ambil semua embedding dari database dan kembalikan dalam bentuk dict"""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.name, f.embedding 
        FROM face_images f 
        JOIN users u ON f.user_id = u.id
    """)
    rows = cur.fetchall()
    conn.close()

    data = {}
    for name, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        if name not in data:
            data[name] = []
        data[name].append(emb)
    return data