import scipy.io
import matplotlib.pyplot as plt

from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
from n_cuts import n_cuts, calculate_n_cut_value
from n_cuts import n_cuts_recursive  # assumed you have this in a separate file

# Φόρτωση εικόνων
data = scipy.io.loadmat(r'c:/Users/nirva/OneDrive/Υπολογιστής/DIP/DIP3/dip_hw_3.mat')

images = {
    "d2a": data["d2a"],
    "d2b": data["d2b"]
}

# Κατώφλια για recursive ncuts
T1 = 5
T2 = 0.20


# Προβολή της αρχικής εικόνας d2a
plt.imshow(images["d2a"], cmap='gray')  # Βάλε cmap='gray' αν είναι ασπρόμαυρη εικόνα
plt.title("Original Image: d2a")
plt.axis('off')
plt.show()

# Αν θέλεις και την d2b:
plt.imshow(images["d2b"], cmap='gray')
plt.title("Original Image: d2b")
plt.axis('off')
plt.show()

for name, img in images.items():
    M, N, C = img.shape
    print(f"\nProcessing {name} with shape {img.shape}")

    affinity = image_to_graph(img)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(rf"Σύγκριση clustering μεθόδων για $\bf{{{name}}}$", fontsize=15)

    # Αναδρομικό n-cuts
    labels_recursive = n_cuts_recursive(affinity, T1=T1, T2=T2)
    axs[0].imshow(labels_recursive.reshape(M, N), cmap="viridis")
    axs[0].set_title(f"Recursive n-cuts\nT1={T1}, T2={T2}")
    axs[0].axis("off")

    # Μη-αναδρομικό n-cuts με k=2
    labels_nc2 = n_cuts(affinity, k=2)
    axs[1].imshow(labels_nc2.reshape(M, N), cmap="viridis")
    axs[1].set_title("n-cuts (k=2)")
    axs[1].axis("off")

    # Μη-αναδρομικό n-cuts με k=3
    labels_nc3 = n_cuts(affinity, k=3)
    axs[2].imshow(labels_nc3.reshape(M, N), cmap="viridis")
    axs[2].set_title("n-cuts (k=3)")
    axs[2].axis("off")

    # Spectral clustering με k=2
    labels_spec = spectral_clustering(affinity, k=2)
    axs[3].imshow(labels_spec.reshape(M, N), cmap="viridis")
    axs[3].set_title("Spectral Clustering (k=2)")
    axs[3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()
