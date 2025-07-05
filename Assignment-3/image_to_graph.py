import numpy as np

def image_to_graph(img_array: np.ndarray) -> np.ndarray:

    M, N, C = img_array.shape

    flat_pixels = img_array.reshape((-1, C))

    # Υπολογισμός Ευκλείδειας απόστασης μεταξύ όλων των pixels
    diff = flat_pixels[:, np.newaxis, :] - flat_pixels[np.newaxis, :, :]  # shape: [MN, MN, C]
    dists = np.linalg.norm(diff, axis=2)  # shape: [MN, MN]

    # Υπολογισμός του affinity matrix σύμφωνα με τον τύπο A(i, j) = 1 / e^{d(i, j)}
    affinity_mat = np.exp(-dists)  # shape: [MN, MN]

    return affinity_mat
#return affinity_mat.astype(float)
