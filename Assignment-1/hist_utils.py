import numpy as np
from typing import Dict

def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict:
    
    # flatten της εικ΄όνας
    flattened_img = img_array.flatten()
    
    # ορισμός του Dictionary
    hist = {}

    # Εύρεση των unique values
    unique_vals = np.unique(flattened_img)

    # Μέτρηση των δειγμάτων για κάθε στάθμη και "γέμισμα" του Dictionary 
    for val in unique_vals:
        count = np.sum(flattened_img == val)
        hist[val] = count

    # Κανονικοποίηση
    if return_normalized:
        total_pixels = flattened_img.size
        for val in hist:
            hist[val] = hist[val] / total_pixels

    return hist


def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict) -> np.ndarray:
    
    processed_img = np.zeros_like(img_array)

    # Εφαρμογή του μεταχηματισμού όπως ορίζεται από το Dictionary 
    for fi, gi in modification_transform.items():
        processed_img[img_array == fi] = gi

    return processed_img

