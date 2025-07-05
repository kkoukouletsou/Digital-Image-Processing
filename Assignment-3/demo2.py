import scipy.io
from spectral_clustering import spectral_clustering
from image_to_graph import image_to_graph
import matplotlib.pyplot as plt

data = scipy.io.loadmat(r'c:/Users/nirva/OneDrive/Υπολογιστής/DIP/DIP3/dip_hw_3.mat')

images = {
    "d2a": data["d2a"],
    "d2b": data["d2b"]
}

for name, img in images.items():
    M, N, C = img.shape
    print(f"Processing {name} with shape {img.shape}")

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"Αρχική εικόνα: {name}")
    plt.axis("off")
    plt.show()
    
    affinity = image_to_graph(img)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(rf"Αποτελέσματα spectral clustering για $\bf{{{name}}}$", fontsize=16)

    for idx, k in enumerate([2, 3, 4]):
        labels = spectral_clustering(affinity, k)
        clustered_img = labels.reshape(M, N)

        ax = axs[idx]
        ax.imshow(clustered_img, cmap="viridis")
        ax.set_title(rf"$k = {k}$", fontsize=12)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    plt.show()
