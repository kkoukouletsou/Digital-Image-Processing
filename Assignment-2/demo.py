from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import sobel_edge, log_edge1, log_edge2, circ_hough
import cv2

# Activate Latex Interpreter 
plt.rcParams['text.usetex'] = True

# Load Image and Convert to Grayscale
img_rgb = Image.open('C:/Users/nirva/OneDrive/Υπολογιστής/DIP/DIP2/basketball_large.png')
img_gray = img_rgb.convert("L")
img_array = np.asarray(img_gray).astype(float) / 255.0

# Plot Sobel Edge Image for different Threshold
thresholds = np.linspace(0.1, 1.0, 10)
sobel_edges = []
num_edges = []

for i in thresholds:
    edge_img = sobel_edge(img_array, thres = i)
    sobel_edges.append(edge_img)
    num_edges.append(np.sum(edge_img))

fig1, axes1 = plt.subplots(2, 5, figsize=(20, 8))
for idx, thres in enumerate(thresholds):
    sobel_img = sobel_edge(img_array, thres=thres)
    row = idx // 5
    col = idx % 5
    ax = axes1[row, col]
    ax.imshow(sobel_img, cmap = 'gray')
    ax.set_title(rf"$\mathrm{{Threshold}} = {thres:.1f}$")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Plot Diagram 
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(thresholds, num_edges, marker='o')
ax2.set_xlabel(r"Threshold")
ax2.set_ylabel(r"Number of Edge Points")
ax2.set_title(r"Sobel Edge: Num of Points vs Threshold")
ax2.grid(True)
plt.show()

# Plot Chosen Sobel Image for Hough Edge Detection 
sobel_img = sobel_edge(img_array, thres=0.4)
fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.imshow(sobel_img, cmap='gray')
ax3.set_title(r"Chosen Sobel Image (Threshold = 0.4)")
ax3.axis('off')
plt.show()


# Plot Thrsehold και Matching Ratio για το LoG Edge Version 1
'''
thresholds = np.round(np.arange(0.5, 1.01, 0.1), 2)
ratios = np.round(np.arange(0.5, 1.01, 0.1), 2)

fig, axes = plt.subplots(len(thresholds), len(ratios), figsize=(20, 20))
fig.suptitle("LOG Edge Detection — Threshold & Match Ratio Sweep", fontsize=16)

for i, t in enumerate(thresholds):
    for j, r in enumerate(ratios):
        result = log_edge(img_array, threshold=t, match_ratio=r)

        ax = axes[i, j]
        ax.imshow(result, cmap='gray')
        ax.set_title(f"T={t}, R={r}")
        ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
'''

# ******************** Choose LoG Edge Version ******************* 
# 1st Version of LoG Edge
# log_img = log_edge1(img_array)  # Uncomment this line if you want to choose Version 1

# 2nd Version of LoG Edge
log_img = log_edge2(img_array, 0.1, 0.15) # Uncomment this line if you want to choose Version 2

# Plot LoG Edge Image 
fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.imshow(log_img, cmap='gray')
ax4.set_title(r"LoG Edge Detection")
ax4.axis('off')
plt.show()

# Define Colors for Plots
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Hough for Sobel 
R_max = 430
dim = np.array([25, 25, 30])
V_min = 22000
plt.figure(figsize=(20, 8))

centers, radii = circ_hough(sobel_img, R_max, dim, V_min)

img_disp = np.asarray(img_rgb).copy()
img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)

for center, radius in zip(centers, radii):
    a, b = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))
    cv2.circle(img_disp, (a, b), r, colors[4], 1)

plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
plt.title(rf"$V_{{\min}} = {V_min}$")
plt.axis('off')
plt.suptitle(r"Circle Detection with Hough (Sobel Edges)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



# Hough for LoG Edge 
R_max = 430
dim = np.array([25, 25, 30])
V_min = 15000 # Change to 40000 for LoG Edge Version 2
plt.figure(figsize=(20, 8))

centers, radii = circ_hough(log_img, R_max, dim, V_min)

img_disp = np.asarray(img_rgb).copy()
img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)

for center, radius in zip(centers, radii):
    a, b = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))
    cv2.circle(img_disp, (a, b), r, colors[4], 1)

plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
plt.title(rf"$V_{{\min}} = {V_min}$")
plt.axis('off')
plt.suptitle(r"Circle Detection with Hough ($\mathrm{LoG}$ Edges)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


