import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform

import numpy as np
from typing import Dict


def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:

    N = img_array.size
    Lg = len(hist_ref)

    # Ορισμός του Dictionary στο οποίο θα αποθηκεύεται η αντιστοίχιση του μετασχηματισμού
    modification_transform = {}

    # Μέθοδος Greedy
    if mode == "greedy":
        # Υπολογισμός αρχικού ιστογράμματος
        hist = calculate_hist_of_img(img_array, return_normalized=False)
        
        # Ταξινόμηση των ιστογραμμάτων με αύξουσα σειρά
        sorted_input_vals = sorted(hist.keys())
        sorted_ref_vals = sorted(hist_ref.keys())

        i = 0

        # Έλεγχος που τρέχει για κάθε στάθμη εξόδου
        for g_val in sorted_ref_vals:
            
            # Αρχικοποίηση του μετρητή count της στάθμης εξόδου σε μηδέν
            count = 0
            # Υπολογισμός του επιθυμητού count της στάθμης
            target_count_per_bin = N * hist_ref[g_val]

            # Όσο το count δεν έχει φτάσει το target
            while i < len(sorted_input_vals) and count < target_count_per_bin:
                fi = sorted_input_vals[i]

                # Αντιστοίχιση της στάθμης f_i στην στάθμη εξόδου g_j
                modification_transform[fi] = g_val 

                # Ανανέωση του count της στάθμης
                count += hist[fi]
                i += 1

        # Αν υπάρχουν στάθμες εισόδου που δεν έχουν αντιστοιχηθεί, να αντιστοιχηθούν στην τελευταία στάθμη εξόδου
        while i < len(sorted_input_vals):
            modification_transform[sorted_input_vals[i]] = sorted_ref_vals[-1]
            i += 1

        processed_img = apply_hist_modification_transform(img_array, modification_transform)
        return processed_img

    # Μέθοδος non-greedy
    if mode == "non-greedy":
        hist = calculate_hist_of_img(img_array, return_normalized=False)
        sorted_input_vals = sorted(hist.keys())
        sorted_ref_vals = sorted(hist_ref.keys())

        i = 0
        for g_val in sorted_ref_vals:
            target_count_per_bin = N * hist_ref[g_val]
            count = 0

            # Κάθε πρώτη αντιστοίχιση του fi στην νέα στάθμη g_j (μπαίνει ΧΩΡΙΣ έλεγχο του deficiency)
            if i < len(sorted_input_vals):
                fi = sorted_input_vals[i]
                modification_transform[fi] = g_val
                count += hist[fi]
                i += 1

            # Για τις επόμενες υποψήφιες αντιστοιχίσεις ξεκινά ο έλεγχος deficiency
            while i < len(sorted_input_vals):
                fi = sorted_input_vals[i]
                deficiency = target_count_per_bin - count

                if deficiency >= hist[fi] / 2:
                    modification_transform[fi] = g_val
                    count += hist[fi]
                    i += 1
                else:
                    break

        #  Ό,τι περισσεύει, μπαίνει στο τελευταίο g_val
        while i < len(sorted_input_vals):
            modification_transform[sorted_input_vals[i]] = sorted_ref_vals[-1]
            i += 1

        processed_img = apply_hist_modification_transform(img_array, modification_transform)
        return processed_img
    
    if mode == "post-disturbance":
    
        # Δημιουργία και εφαρμογή του θορύβου
        d = 1.0 / 255.0
        noise = np.random.uniform(-d/2, d/2, size=img_array.shape)
        post_disturbance_img = img_array + noise
       # post_disturbance_img = np.clip(post_disturbance_img, 0.0, 1.0)


        # Εύρεση των νέων σταθμών εισόδου της f_hat 
        flat_img = post_disturbance_img.flatten()
        unique_vals, counts = np.unique(flat_img, return_counts=True)
 
        '''
        for val, c in zip(unique_vals, counts):
            if c != 1:
                print(f"{val:.5f} appears {c} times")'''

        sorted_indices = np.argsort(unique_vals)
        sorted_vals = unique_vals[sorted_indices]
        sorted_counts = counts[sorted_indices]

        sorted_ref_vals = sorted(hist_ref.keys())
    
        i = 0

        # Για κάθε στάθμη εξόδου
        for g_val in sorted_ref_vals:
            
            # Αρχικοποιείται πάλι σε 0 το count της στάθμης και υπολογίζεται το target
            count = 0
            target_count_per_bin = N * hist_ref[g_val]

            # Όσο το count είναι μικρότερο του target, ανατίθενται και άλλες στάθμες εισόδου και ανανεώνεται το count
            while i < len(sorted_vals) and count < target_count_per_bin:
                fi = sorted_vals[i]
                modification_transform[fi] = g_val
                count += sorted_counts[i]
                i += 1

        # Ό,τι περισσεύει μπαίνει στην τελευταία στάθμη εξόδου
        while i < len(sorted_vals):
            modification_transform[sorted_vals[i]] = sorted_ref_vals[-1]
            i += 1

        processed_img = apply_hist_modification_transform(post_disturbance_img, modification_transform)
        return processed_img

    
def perform_hist_eq(img_array: np.ndarray, mode: str)-> np.ndarray:
    Lg = 256
    
    # Uncomment if you want range g = [0, 1]
    gmin = 0 
    gmax = 1
    
    # Uncomment if you want range g = [fmin, fmax]
    #gmin = np.min(img_array)
    #gmax = np.max(img_array)
    output_levels = np.linspace(gmin, gmax, Lg)

    # ορισμός της ομοιόμορφης κατανομής (επιθυμητή κατανομή)
    hist_ref = {g: 1/Lg for g in output_levels}
    equalized_img = perform_hist_modification(img_array, hist_ref, mode)
    return equalized_img

def perform_hist_matching(img_array, img_array_ref, mode)-> np.ndarray:
    return_normalized = True
    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized)
    processed_img = perform_hist_modification(img_array, hist_ref, mode)
    return processed_img




