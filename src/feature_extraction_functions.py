import cv2
import numpy as np


import pandas as pd
import numpy as np
import cv2
from numba import njit, prange
from joblib import Parallel, delayed
import logging
import math


# --- Setup logging for progress tracking ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# OPTIMIZATION 1: PRE-COMPUTED GLOBALS FOR LTP
# ==============================================================================
LTP_P = 8
LTP_R = 1
FFT_DESCRIPTOR = 11

# Pre-calculate the total number of features for array pre-allocation
NUM_LTP_FEATS_one = 2 * (LTP_P + 2)

# CORRECTED a mismatch between calculated features and this constant.
NUM_SHAPE_FEATS = 29 + FFT_DESCRIPTOR 

TOTAL_FEATURES = NUM_LTP_FEATS_one  + NUM_SHAPE_FEATS 

@njit(fastmath=True)
def compute_ltp_riu2_fast(image: np.ndarray, P: int = None, R: int = None, threshold: int = 5) -> np.ndarray:

    # ==========================================================================
    # Part 1: Internal generation of offsets and RIU2 Look-Up Table (LUT).
    # Numba is smart enough to run this code only once during JIT compilation.
    # ==========================================================================

    # Generate neighbor offsets
    offy = np.empty(P, dtype=np.int32)
    offx = np.empty(P, dtype=np.int32)
    for p in range(P):
        angle = 2.0 * math.pi * p / P
        offy[p] = int(round(-R * math.sin(angle)))
        offx[p] = int(round(R * math.cos(angle)))

    # Generate the Rotation Invariant Uniform 2 (RIU2) Look-Up Table
    lut = np.empty(2**P, dtype=np.int32)
    for i in range(2**P):
        # Count bit transitions in the circular binary pattern `i`
        transitions = 0
        # Check transition between the last and first bit
        if ((i >> (P - 1)) & 1) != (i & 1):
            transitions += 1
        # Check transitions between adjacent bits
        for p in range(P - 1):
            if ((i >> p) & 1) != ((i >> (p + 1)) & 1):
                transitions += 1

        # If the pattern is "uniform" (<=2 transitions), the code is the
        # number of set bits (popcount). Otherwise, it's a single value.
        if transitions <= 2:
            # Efficiently count set bits (popcount)
            val = i
            popcount = 0
            while val > 0:
                val &= (val - 1)
                popcount += 1
            lut[i] = popcount
        else:
            lut[i] = P + 1

    # ==========================================================================
    # Part 2: Main LTP histogram computation using the generated LUT.
    # ==========================================================================
    rows, cols = image.shape
    nr, nc = rows - 2 * R, cols - 2 * R
    if nr <= 0 or nc <= 0:
        return np.zeros(2 * (P + 2), dtype=np.float32)

    ubins = np.zeros(P + 2, dtype=np.int32)
    lbins = np.zeros(P + 2, dtype=np.int32)

    for i in prange(nr):
        for j in prange(nc):
            y, x = i + R, j + R
            c = image[y, x]
            ub_code, lb_code = 0, 0
            for p in prange(P):
                # Calculate difference with neighbor
                diff = image[y + offy[p], x + offx[p]] - c
                # Build the upper and lower binary codes
                if diff > threshold:
                    ub_code |= (1 << p)
                elif diff < -threshold:
                    lb_code |= (1 << p)

            # Use the LUT to find the correct bin and increment
            ubins[lut[ub_code]] += 1
            lbins[lut[lb_code]] += 1

    # ==========================================================================
    # Part 3: Normalize and return the final feature histogram.
    # ==========================================================================
    total = float(nr * nc)
    dist = np.empty(2 * (P + 2), dtype=np.float32)
    if total > 0:
        dist[:P+2] = ubins / total
        dist[P+2:] = lbins / total
    return dist

def get_main_contour(image: np.ndarray):
    """Finds the largest contour in a binary image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None

def _compute_fourier_descriptors(contour, num_descriptors=FFT_DESCRIPTOR):

    if contour.shape[0] < 2:
        return np.zeros(num_descriptors, dtype=np.float32)

    complex_contour = contour[:, 0, 0].astype(np.float64) + 1j * contour[:, 0, 1].astype(np.float64)
    dft = np.fft.fft(complex_contour)

    dft[0] = 0

    # Get the magnitude of the second coefficient for scale normalization
    dft_abs_1 = np.abs(dft[1])
    if dft_abs_1 < 1e-6:
        return np.zeros(num_descriptors, dtype=np.float32)

    # Make descriptors scale-invariant and rotation-invariant (magnitudes)
    magnitudes = np.abs(dft) / dft_abs_1

    # Create a zero array of the final desired size
    descriptors = np.zeros(num_descriptors, dtype=np.float32)
    
    # Get the available coefficients (excluding the DC component)
    source_coeffs = magnitudes[1:]
    
    # Determine how many coefficients to copy (the smaller of available vs. desired)
    n_to_copy = min(num_descriptors, len(source_coeffs))
    
    # Fill the start of the final array with the available coefficients
    descriptors[:n_to_copy] = source_coeffs[:n_to_copy]

    return descriptors

def compute_shape_features_from_contour(main_contour):
    """
    Computes a comprehensive set of shape features from a contour, now including
    robust convexity defect analysis and Fourier Descriptors.

    Gestisce l'errore cv2.error: "convex hull indices are not monotonous" 
    tentando la semplificazione del contorno per prevenire crash in joblib.
    """
    
    # 1. Caso base: contorno non valido
    if main_contour is None or len(main_contour) < 6:
        logging.info("Contorno non valido (None o lunghezza < 6). Ritorno zeri.")
        return np.zeros(NUM_SHAPE_FEATS, dtype=np.float32)

    # L'array di zeri di default in caso di fallimento critico
    zero_feats = np.zeros(NUM_SHAPE_FEATS, dtype=np.float32)
    
    try:
        # --- CALCOLI INIZIALI ---
        moments = cv2.moments(main_contour)
        hu_moments = cv2.HuMoments(moments)
        with np.errstate(divide='ignore', invalid='ignore'):
            hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
            hu_moments_log[np.isinf(hu_moments_log) | np.isnan(hu_moments_log)] = 0
            
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)

        if area <= 0 or perimeter <= 0:
            logging.warning("Area o Perimetro <= 0. Ritorno zeri.")
            return zero_feats

        # --- Standard Physical and Moment Features ---
        m00 = moments.get('m00', 1.0)
        mu20 = moments.get('mu20', 0.0) / m00
        mu02 = moments.get('mu02', 0.0) / m00
        mu11 = moments.get('mu11', 0.0) / m00
        inertia_tensor = np.array([[mu02, -mu11], [-mu11, mu20]])
        eigvals = np.linalg.eigvals(inertia_tensor)
        eigvals = np.sort(eigvals)[::-1]
        inertia_eigval1, inertia_eigval2 = eigvals[0], eigvals[1]
        
        # minAreaRect
        (x_rot, y_rot), (w_rot, h_rot), _ = cv2.minAreaRect(main_contour)
        if w_rot < h_rot: w_rot, h_rot = h_rot, w_rot
        aspect_ratio_rotated = w_rot / h_rot if h_rot > 0 else 0
        extent = area / (w_rot * h_rot) if (w_rot * h_rot) > 0 else 0
        caliper_diameter_proxy = np.sqrt(w_rot**2 + h_rot**2)

        # Ellipse fit
        (_, (minor_axis, major_axis), orientation) = cv2.fitEllipse(main_contour)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2) if major_axis > 0 else 0
        
        # Altre misure standard
        circularity = (4 * np.pi * area) / (perimeter**2)
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        num_vertices = len(cv2.approxPolyDP(main_contour, 0.01 * perimeter, True))
        
        # Bounding Box
        x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(main_contour)
        bbox_aspect_ratio = w_bbox / h_bbox if h_bbox > 0 else 0

        # Punti Estremi
        leftmost = tuple(main_contour[main_contour[:,:,0].argmin()][0])
        rightmost = tuple(main_contour[main_contour[:,:,0].argmax()][0])
        topmost = tuple(main_contour[main_contour[:,:,1].argmin()][0])
        bottommost = tuple(main_contour[main_contour[:,:,1].argmax()][0])
        extreme_width = rightmost[0] - leftmost[0]
        extreme_height = bottommost[1] - topmost[1]

        # --- Convexity and Defect Features (con gestione errore robusta) ---
        hull = cv2.convexHull(main_contour, returnPoints=False)
        hull_pts = main_contour[hull.flatten()]
        hull_area = cv2.contourArea(hull_pts)
        hull_perimeter = cv2.arcLength(hull_pts, True)
        solidity = area / hull_area if hull_area > 0 else 0
        convexity = hull_perimeter / perimeter if perimeter > 0 else 0
        
        # Variabili di default per i difetti
        num_defects = 0
        mean_defect_depth = 0.0
        max_defect_depth = 0.0
        
        defects = None
        
        if len(hull) > 3:
            temp_contour_for_defects = main_contour
            temp_hull_for_defects = hull
            
            try:
                # TENTATIVO 1: Contorno originale
                defects = cv2.convexityDefects(temp_contour_for_defects, temp_hull_for_defects)
            except cv2.error as e:
                # Cattura l'errore specifico di auto-intersezione
                if "convex hull indices are not monotonous" in str(e):
                    logging.warning("ConvexityDefects fallito (auto-intersezione). Tentativo di recupero tramite semplificazione...")
                    
                    # RECUPERO: Semplifica il contorno
                    epsilon = 0.01 * perimeter
                    simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
                    
                    if len(simplified_contour) >= 3:
                        # Ricalcola l'hull sul contorno semplificato
                        temp_hull_for_defects = cv2.convexHull(simplified_contour, returnPoints=False)
                        
                        if len(temp_hull_for_defects) > 3:
                            try:
                                # TENTATIVO 2: Contorno semplificato
                                defects = cv2.convexityDefects(simplified_contour, temp_hull_for_defects)
                                logging.warning(f"Recupero riuscito con contorno semplificato (punti: {len(simplified_contour)}).")
                            except cv2.error:
                                # Fallito anche con il contorno semplificato. Defects rimane None.
                                logging.error("Fallimento anche con contorno semplificato. I difetti saranno impostati a zero.")
                                pass
                        else:
                             logging.warning("Il contorno semplificato è troppo corto per calcolare l'hull.")
                    else:
                         logging.warning("Il contorno semplificato è troppo corto.")
                else:
                    # Se è un errore cv2 diverso da quello atteso, lo rilanciamo per il blocco catch generale
                    raise e
            
            # Se defects è stato calcolato (sia con l'originale che con il semplificato)
            if defects is not None and defects.size > 0:
                num_defects = defects.shape[0]
                # defects[:, 0, 3] è la distanza dal punto del contorno all'hull (scaled by 256)
                defect_depths = defects[:, 0, 3] / 256.0
                mean_defect_depth = np.mean(defect_depths)
                max_defect_depth = np.max(defect_depths)

        # --- Fourier Descriptors ---
        fourier_descriptors = _compute_fourier_descriptors(main_contour) 

        # --- Combina tutte le feature ---
        physical_feats = np.array([
            area, perimeter, aspect_ratio_rotated, extent, solidity,
            circularity,
            equivalent_diameter, eccentricity, num_vertices, major_axis, minor_axis,
            orientation, inertia_eigval1, inertia_eigval2,
            bbox_aspect_ratio, caliper_diameter_proxy, extreme_width, extreme_height,
            convexity, num_defects, mean_defect_depth, max_defect_depth 
        ], dtype=np.float32)

        return np.concatenate([
            hu_moments_log.flatten(),
            fourier_descriptors,
            physical_feats
        ])

    except Exception as e:
        # Cattura qualsiasi altro errore critico (es. fitEllipse, np.linalg.eigvals)
        logging.critical(f"ERRORE CRITICO in compute_shape_features_from_contour: {e}")
        return zero_feats

# ==============================================================================
# OPTIMIZATION 2: BATCH PROCESSING PIPELINE
# ==============================================================================

def _process_batch(image_batch: np.ndarray) -> np.ndarray:
    """Processes a batch of images, returning a pre-allocated batch of feature vectors."""
    feature_batch = np.empty((len(image_batch), TOTAL_FEATURES), dtype=np.float32)

    for i, image in enumerate(image_batch):
        img_contiguous = np.asarray(image, dtype=np.uint8, order="C")

        main_contour = get_main_contour(img_contiguous)
        shape_feats = compute_shape_features_from_contour(main_contour)
        ltp_feats_one = compute_ltp_riu2_fast(img_contiguous, P = LTP_P, R = LTP_R)
      

        #haralick_features = compute_haralick_features(img_contiguous, levels = LEVELS, distances = DISTANCES)
        feature_batch[i] = np.concatenate([ltp_feats_one, shape_feats])

    return feature_batch

def extract_feats(image_df: pd.DataFrame, n_jobs: int = -1, batch_size: int = 128, verbosity: int = 0) -> pd.DataFrame:
    """
    Extracts all features using batch processing for maximum throughput.
    """
    if 'image' not in image_df.columns or 'label' not in image_df.columns:
        raise ValueError("Input DataFrame must contain 'image' and 'label' columns.")

    logging.info(f"Starting optimized feature extraction for {len(image_df)} images with batch size {batch_size}...")

    images_np = image_df['image'].to_numpy()
    num_images = len(images_np)

    batches = [images_np[i:i + batch_size] for i in range(0, num_images, batch_size)]

    feature_batches = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_batch)(batch) for batch in batches
    )

    feature_array = np.vstack(feature_batches)
    if verbosity > 0:
        logging.info(f"Feature extraction complete. Shape of feature array: {feature_array.shape}")

    # --- Create descriptive column names ---
    ltp_names_one = [f'ltp_u_{i}' for i in range(LTP_P+2)] + [f'ltp_l_{i}' for i in range(LTP_P+2)]
    
    hu_names = [f'hu_{i}' for i in range(7)]
    fft_names = [f'fft_{i}' for i in range(FFT_DESCRIPTOR)]
    shape_names = [ 
        'shp_area', 'shp_perimeter', 'shp_aspect_ratio', 'shp_extent',
        'shp_solidity', 'shp_circularity', 'shp_eq_diameter', 'shp_eccentricity',
        'shp_vertices', 'shp_major_axis', 'shp_minor_axis', 'orientation', 'inertia_eigval1', 'inertia_eigval2',
        'bbox_aspect_ratio', 'caliper_diameter_proxy', 'extreme_width', 'extreme_height', 'shp_convexity',
        'shp_num_defects', 'shp_mean_defect_depth', 'shp_max_defect_depth', 
    ]
    

    all_column_names = ltp_names_one   + hu_names  + fft_names +  shape_names 

    features_df = pd.DataFrame(feature_array, columns=all_column_names, index=image_df.index)
    features_df['label'] = image_df['label']

    if verbosity > 0:
        logging.info("Successfully created final features DataFrame.")
    return features_df

