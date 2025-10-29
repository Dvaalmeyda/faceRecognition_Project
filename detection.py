from insightface.app import FaceAnalysis
import cv2

class FaceDetector:
    """
    Class for face detection using InsightFace - https://insightface.readthedocs.io/en
    (buffalo_l)

    Atribut
    --------
    app : FaceAnalysis

    Parameter
    ----------
    det_size : tuple(int, int)
        Defines the size of face detection
    ctx_id : int
        Device id - set as -1 if using CPU
    """

    def __init__(self, det_size=(640, 640), ctx_id=-1):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detectFaces(self, image):
        """
        Parameter
        ----------
        image : numpy.ndarray
           Input image in BGR format (OpenCV)

        Return
        -------
        list : list of face detection
        """
        faces = self.app.get(image)
        return faces

    def drawBoxes(self, image, faces, show=False, save_path=None):
        """
        Menggambar kotak di sekitar wajah yang terdeteksi.

        Parameter
        ----------
        image : numpy.ndarray
            Gambar input tempat kotak wajah akan digambar.
        faces : list
            Daftar wajah hasil deteksi dari fungsi detectFaces().
        show : bool, opsional
            Jika True, tampilkan hasilnya dengan cv2.imshow().
        save_path : str, opsional
            Jika diberikan, hasil gambar disimpan ke path tersebut.

        Return
        -------
        numpy.ndarray
            Gambar dengan kotak wajah yang sudah digambar.
        """
        if not faces:
            print("[WARN] No faces to draw.")
            return image

        # Gambar kotak di setiap wajah
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Tambahkan teks kecil di atas kotak
            cv2.putText(image, "Face", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Jika ingin tampilkan
        if show:
            cv2.imshow("Detected Faces", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Jika ingin disimpan
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"[INFO] Saved image with boxes to: {save_path}")
        return image