from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from hist_modif import perform_hist_eq, perform_hist_matching

# Φόρτωση input και reference image
filename1 = "input_img.jpg"
filename2 = "ref_img.png"

#input_img = Image.open(fp=filename1)
input_img = Image.open(fp='C:/Users/nirva/OneDrive/Υπολογιστής/DIP/input_img.jpg')
#ref_img = Image.open(fp=filename2)
ref_img = Image.open(fp='C:/Users/nirva/OneDrive/Υπολογιστής/DIP/ref_img.jpg')

# Μετατροπή της input και reference εικόνων σε Grayscale
input_img = input_img.convert("L")
ref_img = ref_img.convert("L") # Απαραίτητο παρόλο που είναι ήδη ασπρόμαυρη γιατί η PIL θεωρεί πως έχει 3 κανάλια αλλιώς

input_img.show()
ref_img.show()

# Μετατροπή των εικόνων σε numpy array, με τιμές στο [0,1]
input_img_array = np.array(input_img).astype(float) / 255.0
ref_img_array = np.array(ref_img).astype(float)/255.0

# Υπολογισμός ελάχιστης και μέγιστης τιμής φωτεινότητας
min_val = np.min(input_img)
max_val = np.max(input_img)
print(f"Minimum luminance value: {min_val}")
print(f"Maximum luminance value: {max_val}")


def plot_histogram(img, title="Histogram", color='blue', Lg=256):
    img_flat = img.flatten()

    # Uncomment για να τρέξει εξισορρόπηση με εύρος [fmin, fmax]
    #img_min = img_flat.min()
    #img_max = img_flat.max()
    #bin_edges = np.linspace(img_min, img_max, Lg + 1)

    # Uncomment για να τρέξει εξισορρόπηση με εύρος 0-1 
    bin_edges = np.linspace(0, 1, Lg + 1)

    plt.hist(img_flat, bins=bin_edges, color=color, edgecolor='black', linewidth=0.5)
    plt.title(title)
    plt.xlabel("Φωτεινότητα")
    plt.ylabel("Πλήθος Pixels")
    plt.grid(True)

# Παρουσίαση του ιστογράμματος της input εικόνας
plt.figure(figsize=(8, 5))
plot_histogram(input_img_array, title="Histogram of Grayscale Input Image")
plt.show()

# Παρουσίαση του ιστογράμματος της reference εικόνας
plt.figure(figsize=(8, 5))
plot_histogram(ref_img_array, title="Histogram of Grayscale Reference Image")
plt.show()

# Κλήση Histogram Equalization
# Μέθοδος Greedy
mode = "greedy"
print("Histogram Equalization - Greedy Mode")
equalized_img_greedy = perform_hist_eq(input_img_array, mode)

# Μέθοδος Non Greedy
mode = "non-greedy"
print("Histogram Equalization - Non Greedy Mode")
equalized_img_nongreedy = perform_hist_eq(input_img_array, mode)

# Μέθοδος Post Disturbance
mode = "post-disturbance"
print("Histogram Equalization - Post Disturbance Mode")
equalized_img_postdisturbance = perform_hist_eq(input_img_array, mode)

# Παρουσίαση των εξισορροπημένων εικόνων και των ιστογραμμάτων τους
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(equalized_img_greedy, cmap='gray')
plt.title("Histogram Equalization (Greedy)")
plt.subplot(1, 2, 2)
plot_histogram(equalized_img_greedy, title="Histogram (Greedy)", color='green')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(equalized_img_nongreedy, cmap='gray')
plt.title("Histogram Equalization (Non-Greedy)")
plt.subplot(1, 2, 2)
plot_histogram(equalized_img_nongreedy, title="Histogram (Non-Greedy)", color='orange')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(equalized_img_postdisturbance, cmap='gray')
plt.title("Histogram Equalization (Post-Disturbance)")
plt.subplot(1, 2, 2)
plot_histogram(equalized_img_postdisturbance, title="Histogram (Post-Disturbance)", color='red')
plt.show()

# Κλήση Histogram Matching
# Μέθοδος Greedy
mode = "greedy"
print("Histogram Matching - Greedy Mode")
processed_img_greedy = perform_hist_matching(input_img_array, ref_img_array, mode)

# Μέθοδος Non Greedy
mode = "non-greedy"
print("Histogram Matching - Non Greedy Mode")
processed_img_nongreedy = perform_hist_matching(input_img_array, ref_img_array, mode)

# Μέθοδος Post Disturbance
mode = "post-disturbance"
print("Histogram Matching - Post Disturbance Mode")
processed_img_postdisturbance = perform_hist_matching(input_img_array, ref_img_array, mode)

# Παρουσίαση των εικόνων μετά την αντιστοίχιση ιστογράμματος και των ιστογραμμάτων τους
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(processed_img_greedy, cmap='gray')
plt.title("Histogram Matching (Greedy)")
plt.subplot(1, 2, 2)
plot_histogram(processed_img_greedy, title="Histogram (Greedy)", color='green')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(processed_img_nongreedy, cmap='gray')
plt.title("Histogram Matching (Non-Greedy)")
plt.subplot(1, 2, 2)
plot_histogram(processed_img_nongreedy, title="Histogram (Non-Greedy)", color='orange')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(processed_img_postdisturbance, cmap='gray')
plt.title("Histogram Matching (Post-Disturbance)")
plt.subplot(1, 2, 2)
plot_histogram(processed_img_postdisturbance, title="Histogram (Post-Disturbance)", color='red')
plt.show()

