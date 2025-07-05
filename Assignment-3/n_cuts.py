import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    # D: Διαγώνιος πίνακας βαθμών
    D = np.diag(np.sum(affinity_mat, axis=1))
    # L = D - W
    L = D - affinity_mat
    # Λύση του γενικευμένου προβλήματος ιδιοτιμών: Lx = λDx
    eigvals, eigvecs = eigsh(L, k=k, M=D, which='SM')
    # Κατασκευή πίνακα U
    U = eigvecs
    # Εφαρμογή k-means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1) 
    cluster_idx = kmeans.fit_predict(U)
    return cluster_idx.astype(float)




def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:
    A = np.where(cluster_idx == 0)[0]
    B = np.where(cluster_idx == 1)[0]
    V = np.arange(len(cluster_idx))

    assoc = lambda X, Y: np.sum(affinity_mat[np.ix_(X, Y)])

    assoc_AA = assoc(A, A)
    assoc_AV = assoc(A, V)
    assoc_BB = assoc(B, B)
    assoc_BV = assoc(B, V)

    Nassoc = assoc_AA / assoc_AV + assoc_BB / assoc_BV
    Ncut = 2 - Nassoc

    return Ncut


def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    n = affinity_mat.shape[0]
    cluster_labels = np.full(n, -1, dtype=float)
    cluster_counter = [0]

    def recursive_split(indices):
        if len(indices) <= T1:
            cluster_labels[indices] = cluster_counter[0]
            cluster_counter[0] += 1
            return

        sub_affinity = affinity_mat[np.ix_(indices, indices)]
        sub_labels = n_cuts(sub_affinity, k=2)

        ncut_val = calculate_n_cut_value(sub_affinity, sub_labels)

        if ncut_val > T2:
            cluster_labels[indices] = cluster_counter[0]
            cluster_counter[0] += 1
            return

        idx_A = indices[sub_labels == 0]
        idx_B = indices[sub_labels == 1]

        if len(idx_A) <= T1 or len(idx_B) <= T1:
            cluster_labels[indices] = cluster_counter[0]
            cluster_counter[0] += 1
            return

        recursive_split(idx_A)
        recursive_split(idx_B)

    recursive_split(np.arange(n))
    return cluster_labels

