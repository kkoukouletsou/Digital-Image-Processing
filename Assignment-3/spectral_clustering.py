import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh

def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    # Υπολογισμός του διαγώνιου D
    D = np.diag(np.sum(affinity_mat, axis=1))

    # Υπολογισμός του Λαπλασιανού
    L = D - affinity_mat

    # Υπολογισμός των k μικρότερων ιδιοδιανυσμάτων
    eigvals, eigvecs = eigsh(L, k=k, which='SM')

    # Δημιουργία του πίνακα U
    U = eigvecs  # shape: [n, k]

    # Χρ΄ήση του k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U)
    labels = kmeans.labels_

    return labels.astype(float)

