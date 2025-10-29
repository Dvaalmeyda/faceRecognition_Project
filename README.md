# Face Recognition - InsightFace

## Overview

This is a mini project implementing **InsightFace** for face recognition using **FastAPI** and **PostgreSQL**.

---

## Features

* Register new users and their face images
* Detect faces from input images
* Match faces against the database
* Store face data in PostgreSQL
* Delete face data from the database

---

## Tutorial

1. **Clone the repository:**

```bash
git clone <repo_url>
cd faceRecognition_Project
```

2. **Create a virtual environment:**

```bash
python -m venv .venv
.venv/Scripts/activate
```

3. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Create folder for reference images:**

```bash
mkdir -p resources/references
```

5. **Ensure PostgreSQL is running and create the database:**

```sql
CREATE DATABASE facerec_db;
```

---

## Database Configuration

Update the database connection in `app.py`:

```python
conn = psycopg2.connect(
    host="localhost",
    database="facerec_db",
    user="postgres",
    password="password"
)
```

---

## Running the Server

```bash
python -m uvicorn app:app --reload --workers 1
```

Server will run at: `http://127.0.0.1:8000`
Recommended to open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints

### 1. Register Face

```
POST /api/face/register
```

**Form-data:**

* `file` : image file (`.jpg` / `.png`)
* `name` : string

**Response:**

```json
{
  "user_id": 1,
  "face_id": 3,
  "name": "Alice",
  "message": "Face registered successfully"
}
```

---

### 2. Recognize Face

```
POST /api/face/recognize
```

**Form-data:**

* `file` : image file (`.jpg` / `.png`)

**Response:**

```json
{
  "user_id": 1,
  "name": "Alice",
  "distance": 0.3456,
  "status": "Same Person"
}
```

---

### 3. Get All Faces

```
GET /api/face
```

**Response:**

```json
{
  "faces": [
    {
      "user_id": 1,
      "name": "Alice",
      "face_id": 3,
      "image_path": "resources/references/xxxx.jpg"
    }
  ]
}
```

---

### 4. Delete Face

```
DELETE /api/face/{face_id}
```

**Response:**

```json
{
  "message": "Face 3 deleted successfully"
}
```

---

## Notes

| Field  | Type   | Description                  |
| :----- | :----- | :--------------------------- |
| `file` | `file` | **Required.** `jpg/jpeg/png` |