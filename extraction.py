import numpy as np

class FaceEmbedder:
    def __init__(self, app):
        # Pakai app sama agar tidak muat ulang
        self.app = app

    def getEmbedding(self, face):
        """
        Retrieves and normalizes the embedding vector for a detected face.

        Parameters
        ----------
        face : insightface.model_zoo.arcface.ArcFace
            The detected face object that contains an embedding attribute.

        Returns
        -------
        numpy.ndarray
            A normalized embedding vector representing the face features.
        """
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)  # Normalisasi vektor
        return emb