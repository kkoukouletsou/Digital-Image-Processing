import numpy as np

def fir_conv(in_img_array: np.ndarray, h: np.ndarray, in_origin: np.ndarray, mask_origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Διαστάσεις Εικόνας και Μάσκας
    H, W = in_img_array.shape
    Mh, Mw = h.shape

    # Αντιστροφή Μάσκας
    h_flipped = np.flip(h)

    # Υπολογισμός του Ζero-Padding
    pad_top = mask_origin[0]
    pad_left = mask_origin[1]
    pad_bottom = Mh - pad_top - 1
    pad_right = Mw - pad_left - 1

    padded_img = np.pad(
        in_img_array,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0.0
    )

    # Δημιουργία Εικόνας Εξόδου
    out_img_array = np.zeros_like(in_img_array, dtype=float)

    # Υπολογισμός Συνέλιξης 
    for i in range(H):
        for j in range(W):
            region = padded_img[i:i+Mh, j:j+Mw]
            out_img_array[i, j] = np.sum(region * h_flipped)

    # Νέα Αρχή Εξόδου
    out_origin = in_origin + mask_origin
    return out_img_array, out_origin
   

def sobel_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:

    # Ορισμός των τελεστών Sobel για τις κατευθύνσεις x1 και x2
    Gx1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Gx2 = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1, -2, -1]], dtype=float)
    
    # Κλήση της fir conv για την εφαρμογή της συνέλιξης
    grad_x1, _ = fir_conv(in_img_array, Gx1, np.array([0, 0]), np.array([1, 1]))
    grad_x2, _ = fir_conv(in_img_array, Gx2, np.array([0, 0]), np.array([1, 1]))
    
    # Υπολογισμός του μέτρου
    grad_mag = np.sqrt(grad_x1**2 + grad_x2**2)
    
    # Δημιουργία της δυαδικής εικόνας εξόδου με βάση το κατώφλι
    out_img_array = (grad_mag > thres).astype(int)
    
    return out_img_array

def log_edge2(in_img_array: np.ndarray, threshold: float = 0.6, lum_thresh: float = 0.2) -> np.ndarray:
    
    h = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]
    ], dtype=float)

    lum = in_img_array.astype(float)

    # Συνελικτική συνέλιξη με μικρότερο φίλτρο
    conv_result, _ = fir_conv(in_img_array, h, np.array([0, 0]), np.array([1, 1]))
    H, W = conv_result.shape
    out_img = np.zeros((H, W), dtype=int)

    sign_map = (conv_result >= 0).astype(int)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            region_sign = sign_map[i - 1:i + 2, j - 1:j + 2]
            region_vals = conv_result[i - 1:i + 2, j - 1:j + 2]
            region_lum = lum[i - 1:i + 2, j - 1:j + 2]

            # Κριτήριο αλλαγής πρόσημου (π.χ. 0 και 1 συνυπάρχουν)
            if np.any(region_sign != region_sign[1, 1]):
                max_diff = np.max(region_vals) - np.min(region_vals)
                lum_diff = np.max(region_lum) - np.min(region_lum)

                if max_diff >= threshold and lum_diff >= lum_thresh:
                    out_img[i, j] = 1

    return out_img

def generate_rotations(pattern):
    return [np.rot90(pattern, k) for k in range(4)]

def log_edge1(in_img_array: np.ndarray, threshold: float = 0.6, match_ratio: float = 0.8) -> np.ndarray:
    
    h = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]
    ], dtype=float)

    conv_result, _ = fir_conv(in_img_array, h, np.array([0, 0]), np.array([2, 2]))
    H, W = conv_result.shape
    out_img = np.zeros((H, W), dtype=int)

    # Δυαδικός χάρτης πρόσημου
    sign_map = (conv_result >= 0).astype(int)

    # Αντιπροσωπευτικά patterns 5x5
    base_patterns = [

    np.array([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ]),
    
    np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]),
    
    np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]),
    
    np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1]
        ]),
    
    np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0]
        ]),
    
    np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]),
    
    np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]),
  
    np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]),
        
    np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0]
        ]),
    np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ]),
    np.array([
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ]),
        
    np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]),
        
    np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1]
        ]),
    np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ]),
    np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]),
        
    np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    ]

    # Παράγωγα patterns με περιστροφές
    all_patterns = []
    for pattern in base_patterns:
        all_patterns.extend(generate_rotations(pattern))

    for i in range(2, H - 2):
        for j in range(2, W - 2):
            region_sign = sign_map[i - 2:i + 3, j - 2:j + 3]
            region_vals = conv_result[i - 2:i + 3, j - 2:j + 3]

            max_diff = np.max(region_vals) - np.min(region_vals)
            if max_diff < threshold:
                continue

            for pattern in all_patterns:
                matches = np.sum(region_sign == pattern)
                if matches >= match_ratio * 25:
                    out_img[i, j] = 1
                    break

    return out_img

def circ_hough(in_img_array, R_max, dim, V_min, r_limit=10):
    H, W = in_img_array.shape
    n_a, n_b, n_r = dim

    a_scale = W / n_a
    b_scale = H / n_b

    # Δημιουργία πλέγματος τιμών ακτίνας    
    r_vals = np.linspace(390, R_max, n_r)
    accumulator = np.zeros((n_a, n_b, n_r), dtype=int)

    edge_points = np.argwhere(in_img_array == 1)  # ακμές

    for idx, (y, x) in enumerate(edge_points):
        if idx % 100 == 0:
            print(f"Processing edge point {idx}/{len(edge_points)}")

        for r_idx, r in enumerate(r_vals):
            if r < r_limit:
                continue

            for theta in np.linspace(0, 2 * np.pi, 100, endpoint=False):
                a = x - r * np.cos(theta)
                b = y - r * np.sin(theta)

                a_idx = int(np.round(a / a_scale))
                b_idx = int(np.round(b / b_scale))

                if 0 <= a_idx < n_a and 0 <= b_idx < n_b:
                    accumulator[a_idx, b_idx, r_idx] += 1

    # Ανάκτηση κύκλων με ψήφους > V_min
    centers = []
    radii = []
    for a_idx in range(n_a):
        for b_idx in range(n_b):
            for r_idx in range(n_r):
                if accumulator[a_idx, b_idx, r_idx] >= V_min:
                    a = a_idx * a_scale
                    b = b_idx * b_scale
                    r = r_vals[r_idx]
                    centers.append([a, b])
                    radii.append(r)

    print("Max votes in accumulator:", np.max(accumulator))

    return np.array(centers), np.array(radii)
