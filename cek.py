import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="facerec_db",
    user="postgres",
    password="1234"
)
cur = conn.cursor()
cur.execute("SELECT u.name, f.image_path FROM face_images f JOIN users u ON f.user_id = u.id")
rows = cur.fetchall()
for row in rows:
    print(row)
cur.close()
conn.close()