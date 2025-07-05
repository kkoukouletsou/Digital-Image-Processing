import scipy.io
from spectral_clustering import spectral_clustering


mat = scipy.io.loadmat(r'c:/Users/nirva/OneDrive/Υπολογιστής/DIP/DIP3/dip_hw_3.mat')
#mat = scipy.io.loadmat('dip_hw_3.mat')

affinity = mat['d1a']

for k in [2, 3, 4]:
    labels = spectral_clustering(affinity, k)
    print(f"Spectral clustering results for k={k}:\n", labels.reshape(-1))

