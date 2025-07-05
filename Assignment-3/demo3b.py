import scipy.io
import matplotlib.pyplot as plt
import numpy as np

from n_cuts import n_cuts_recursive, calculate_n_cut_value
from image_to_graph import image_to_graph

data = scipy.io.loadmat(r'c:/Users/nirva/OneDrive/Υπολογιστής/DIP/DIP3/dip_hw_3.mat')

images = {"d2a": data["d2a"], "d2b": data["d2b"]}

for name, img in images.items():
    M, N, C = img.shape
    print(f"\nProcessing {name} with shape {img.shape}")

    affinity = image_to_graph(img)

    # Εκτέλεση αναδρομικής n-cuts για 1 split
    T1 = 500  # threshold για ελάχιστο μέγεθος cluster
    T2 = 1   # threshold για ncut value (πολύ αυστηρό για να σταματήσει μετά το πρώτο)

    labels = n_cuts_recursive(affinity, T1, T2)

    unique_labels = np.unique(labels)
    if len(unique_labels) > 2:
        print(f"Warning: Found more than 2 clusters! Change values of T1 ans T2")

    cluster_idx = (labels == 0).astype(int)

    ncut_val = calculate_n_cut_value(affinity, cluster_idx)
    print(f"Ncut value for {name}: {ncut_val:.4f}")

    segmented_img = labels.reshape(M, N)

    plt.figure(figsize=(6, 5))
    plt.imshow(segmented_img, cmap="viridis")
    plt.title(rf"Recursive $n$-cuts (1 step) for $\bf{{{name}}}$" + f"\nNcut = {ncut_val:.4f}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

