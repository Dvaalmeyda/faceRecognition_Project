import numpy as np

class FaceMatcher:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def compare(self, emb1, emb2):
        dist = np.linalg.norm(emb1 - emb2)
        match = dist < self.threshold
        return dist, match

    def getMatching(self, input_emb, reference_embeddings, verbose=False):
        if not reference_embeddings:
            print("[ERROR] No reference embeddings provided.")
            return None, None

        distances = {name: np.linalg.norm(input_emb - emb) for name, emb in reference_embeddings.items()}

        # Optional logging of all distances
        if verbose:
            print("\n[INFO] Distance to each reference:")
            for name, dist in distances.items():
                print(f"  {name}: {dist:.4f}")

        # Find the closest match
        best_name = min(distances, key=distances.get)
        best_distance = distances[best_name]

        return best_name, best_distance