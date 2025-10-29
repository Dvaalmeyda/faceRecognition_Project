import os
import cv2

def loadReference(folder):
    """
    Load reference images from folder
    :param folder: str
    :return: list of images
    """
    refs = []
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"[INFO] Loaded reference image: {file}")
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is not None:
                refs.append((file, img))
            else:
                print(f"[WARN] Failed to read image: {file}")
    return refs