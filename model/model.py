import urllib.request

# URL model
arcface_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100.onnx"
retinaface_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/retinaface_r50.onnx"

# Dir
arcface_path = "model/arcface_r100.onnx"
retinaface_path = "model/retinaface_r50.onnx"

print("Downloading ArcFace model...")
urllib.request.urlretrieve(arcface_url, arcface_path)
print("ArcFace downloaded successfully.")

print("Downloading RetinaFace model...")
urllib.request.urlretrieve(retinaface_url, retinaface_path)
print("RetinaFace downloaded successfully.")