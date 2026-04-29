import numpy as np
import cv2
import math
import joblib
import gc
from pathlib import Path
from numba import njit

# ==============================================================================
# 1. CORE MATH KERNELS (ZERO COPY)
# ==============================================================================

LTP_P = 8
LTP_R = 1
FFT_DESCRIPTOR = 11

@njit(cache=True, fastmath=True, nogil=True)
def precompute_ltp_constants(P, R):
    offy = np.empty(P, dtype=np.int32)
    offx = np.empty(P, dtype=np.int32)
    offy[0] = 0; offx[0] = 1
    offy[1] = -1; offx[1] = 1
    offy[2] = -1; offx[2] = 0
    offy[3] = -1; offx[3] = -1
    offy[4] = 0; offx[4] = -1
    offy[5] = 1; offx[5] = -1
    offy[6] = 1; offx[6] = 0
    offy[7] = 1; offx[7] = 1

    lut = np.empty(256, dtype=np.int32)
    for i in range(256):
        transitions = 0
        if ((i >> 7) & 1) != (i & 1): transitions += 1
        for p in range(7):
            if ((i >> p) & 1) != ((i >> (p + 1)) & 1): transitions += 1

        if transitions <= 2:
            val = i
            popcount = 0
            while val > 0:
                val &= (val - 1)
                popcount += 1
            lut[i] = popcount
        else:
            lut[i] = 9
    return offy, offx, lut

GLOBAL_OFFY, GLOBAL_OFFX, GLOBAL_LUT = precompute_ltp_constants(LTP_P, LTP_R)

@njit(fastmath=True, cache=True, nogil=True)
def compute_ltp_riu2_inplace(image, offy, offx, lut, out_buffer, start_idx):
    rows, cols = image.shape
    nr, nc = rows - 2, cols - 2
    if nr <= 0 or nc <= 0: return

    ubins = np.zeros(10, dtype=np.int32)
    lbins = np.zeros(10, dtype=np.int32)
    inv_total = 1.0 / float(nr * nc)

    for i in range(nr):
        y = i + 1
        for j in range(nc):
            x = j + 1
            c = int(image[y, x])
            ub_code = 0; lb_code = 0
            
            d = int(image[y, x+1]) - c; 
            if d > 5: ub_code |= 1
            elif d < -5: lb_code |= 1
            
            d = int(image[y-1, x+1]) - c; 
            if d > 5: ub_code |= 2
            elif d < -5: lb_code |= 2
            
            d = int(image[y-1, x]) - c; 
            if d > 5: ub_code |= 4
            elif d < -5: lb_code |= 4
            
            d = int(image[y-1, x-1]) - c; 
            if d > 5: ub_code |= 8
            elif d < -5: lb_code |= 8
            
            d = int(image[y, x-1]) - c; 
            if d > 5: ub_code |= 16
            elif d < -5: lb_code |= 16
            
            d = int(image[y+1, x-1]) - c; 
            if d > 5: ub_code |= 32
            elif d < -5: lb_code |= 32
            
            d = int(image[y+1, x]) - c; 
            if d > 5: ub_code |= 64
            elif d < -5: lb_code |= 64
            
            d = int(image[y+1, x+1]) - c; 
            if d > 5: ub_code |= 128
            elif d < -5: lb_code |= 128
            
            ubins[lut[ub_code]] += 1
            lbins[lut[lb_code]] += 1

    for i in range(10):
        out_buffer[start_idx + i] = ubins[i] * inv_total
        out_buffer[start_idx + 10 + i] = lbins[i] * inv_total

@njit(cache=True, fastmath=True)
def extract_contour_features_numba(contour_x, contour_y):
    n = len(contour_x)
    if n < 4: return 0.0, 0.0, 0.0, 0.0
    
    min_x = contour_x[0]; max_x = contour_x[0]
    min_y = contour_y[0]; max_y = contour_y[0]
    sum_x = 0.0; sum_y = 0.0
    
    for i in range(n):
        x = contour_x[i]; y = contour_y[i]
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
        sum_x += x; sum_y += y
    
    return (max_x - min_x, max_y - min_y, sum_x / n, sum_y / n)

def get_main_contour(image):
    # Get contours and hierarchy
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        del contours, hierarchy
        return None
        
    # Manual Argmax to avoid creating intermediate list of areas
    best_cnt = None
    max_area = -1.0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            best_cnt = c
            
    # CRITICAL: Delete the list to free memory immediately
    del contours
    del hierarchy
    
    return best_cnt

def write_fourier_descriptors_inplace(contour, out_buffer, start_idx, num_descriptors=FFT_DESCRIPTOR):
    if contour.shape[0] < 2: return
    pts = contour[:, 0, :].astype(np.float32)
    complex_contour = pts[:, 0] + 1j * pts[:, 1]
    dft = np.fft.fft(complex_contour)
    dft[0] = 0
    abs_1 = abs(dft[1])
    if abs_1 < 1e-6: return
    
    inv_abs_1 = 1.0 / abs_1
    n = min(num_descriptors, len(dft) - 1)
    out_buffer[start_idx : start_idx + n] = np.abs(dft[1:n+1]) * inv_abs_1

def compute_shape_features_inplace(main_contour, out_buffer, start_idx):
    if main_contour is None or len(main_contour) < 6: return
    try:
        m = cv2.moments(main_contour)
        area = m['m00']
        if area < 1.0: return

        hu = cv2.HuMoments(m).flatten()
        for k in range(7):
            val = hu[k]
            if val != 0: out_buffer[start_idx + k] = -1 * math.copysign(1.0, val) * math.log10(abs(val))
            else: out_buffer[start_idx + k] = 0.0

        current_idx = start_idx + 7
        write_fourier_descriptors_inplace(main_contour, out_buffer, current_idx)
        current_idx += FFT_DESCRIPTOR

        inv_area = 1.0 / area
        perimeter = cv2.arcLength(main_contour, True)
        if perimeter < 1e-6: return
        
        inv_perim = 1.0 / perimeter
        inv_perim_sq = inv_perim * inv_perim
        
        mu20 = m['mu20'] * inv_area; mu02 = m['mu02'] * inv_area; mu11 = m['mu11'] * inv_area
        delta = math.sqrt(4 * mu11*mu11 + (mu20 - mu02)*(mu20 - mu02))
        val1 = (mu20 + mu02 + delta) * 0.5
        val2 = (mu20 + mu02 - delta) * 0.5
        
        rect = cv2.minAreaRect(main_contour)
        (xr, yr), (wr, hr), angle = rect
        if wr < hr: wr, hr = hr, wr
        inv_hr = 1.0 / hr if hr > 1e-6 else 0.0
        ar_rot = wr * inv_hr
        area_rect = wr * hr
        extent = area / area_rect if area_rect > 0 else 0
        caliper = math.sqrt(wr*wr + hr*hr)
        
        ellipse = cv2.fitEllipse(main_contour)
        (_, (ax_min, ax_maj), orient) = ellipse
        ecc = math.sqrt(1.0 - (ax_min/ax_maj)**2) if ax_maj > 1e-6 else 0
        
        circ = (4 * 3.14159265359 * area) * inv_perim_sq
        eq_diam = math.sqrt(4 * area / 3.14159265359)
        
        epsilon = 0.01 * perimeter
        poly = cv2.approxPolyDP(main_contour, epsilon, True)
        n_verts = len(poly)
        
        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(main_contour)
        ar_bbox = float(w_bb) / float(h_bb) if h_bb > 0 else 0
        
        contour_flat = main_contour[:, 0, :]
        ext_w, ext_h, _, _ = extract_contour_features_numba(contour_flat[:, 0], contour_flat[:, 1])
        
        hull = cv2.convexHull(main_contour, returnPoints=False)
        solidity, convexity, n_defects, mean_depth, max_depth = 0.0, 0.0, 0, 0.0, 0.0
        
        if len(hull) > 2:
            hull_pts = main_contour[hull.flatten()]
            h_area = cv2.contourArea(hull_pts)
            h_perim = cv2.arcLength(hull_pts, True)
            solidity = area / h_area if h_area > 0 else 0
            convexity = h_perim * inv_perim
            
            if len(hull) > 3:
                try:
                    defects = cv2.convexityDefects(main_contour, hull)
                    if defects is not None and len(defects) > 0:
                        n_defects = len(defects)
                        depths = defects[:, 0, 3] * 0.00390625
                        mean_depth = depths.mean()
                        max_depth = depths.max()
                except: pass
        
        physical_start = current_idx
        out_buffer[physical_start] = area
        out_buffer[physical_start+1] = perimeter
        out_buffer[physical_start+2] = ar_rot
        out_buffer[physical_start+3] = extent
        out_buffer[physical_start+4] = solidity
        out_buffer[physical_start+5] = circ
        out_buffer[physical_start+6] = eq_diam
        out_buffer[physical_start+7] = ecc
        out_buffer[physical_start+8] = n_verts
        out_buffer[physical_start+9] = ax_maj
        out_buffer[physical_start+10] = ax_min
        out_buffer[physical_start+11] = orient
        out_buffer[physical_start+12] = val1
        out_buffer[physical_start+13] = val2
        out_buffer[physical_start+14] = ar_bbox
        out_buffer[physical_start+15] = caliper
        out_buffer[physical_start+16] = ext_w
        out_buffer[physical_start+17] = ext_h
        out_buffer[physical_start+18] = convexity
        out_buffer[physical_start+19] = n_defects
        out_buffer[physical_start+20] = mean_depth
        out_buffer[physical_start+21] = max_depth
    except: pass

# ==============================================================================
# 2. FEATURE EXTRACTOR CLASS
# ==============================================================================
class RealTimeFeatureExtractor:
    def __init__(self):
        self.LTP_P = LTP_P
        self.LTP_R = LTP_R
        self.ltp_len = 20
        self.total_feats = self.ltp_len + 29 + FFT_DESCRIPTOR
        self.result_buffer = np.zeros((1, self.total_feats), dtype=np.float32) # 2D for easier use
        self.offy = GLOBAL_OFFY; self.offx = GLOBAL_OFFX; self.lut = GLOBAL_LUT
        
        # Name mapping setup (omitted for brevity, same as before)
        self.name_to_idx = {} # ... (ensure this is populated if needed by the factory)
        # Note: If the Factory uses name_to_idx, ensure the init code from previous steps is here.
        # For safety, I will include the minimal init for name_to_idx:
        ltp_names = [f'ltp_u_{i}' for i in range(10)] + [f'ltp_l_{i}' for i in range(10)]
        hu_names = [f'hu_{i}' for i in range(7)]
        fft_names = [f'fft_{i}' for i in range(FFT_DESCRIPTOR)]
        shape_names = [
            'shp_area', 'shp_perimeter', 'shp_aspect_ratio', 'shp_extent',
            'shp_solidity', 'shp_circularity', 'shp_eq_diameter',
            'shp_eccentricity', 'shp_vertices', 'shp_major_axis',
            'shp_minor_axis', 'orientation', 'inertia_eigval1',
            'inertia_eigval2', 'bbox_aspect_ratio',
            'caliper_diameter_proxy', 'extreme_width',
            'extreme_height', 'shp_convexity',
            'shp_num_defects', 'shp_mean_defect_depth',
            'shp_max_defect_depth'
        ]
        feature_names = ltp_names + hu_names + fft_names + shape_names
        self.name_to_idx = {name: i for i, name in enumerate(feature_names)}

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        # Pass 1D view of the buffer
        buffer_view = self.result_buffer[0]
        compute_ltp_riu2_inplace(image, self.offy, self.offx, self.lut, buffer_view, 0)
        cnt = get_main_contour(image)
        if cnt is not None:
            compute_shape_features_inplace(cnt, buffer_view, self.ltp_len)
        else:
            buffer_view[self.ltp_len:] = 0
        return self.result_buffer

# ==============================================================================
# 3. NUCLEAR FAST CLASSIFIER (Final Production Version)
# ==============================================================================

class NuclearFastEnsembleClassifier:
    def __init__(self, asset_folder):
        self.extractor = RealTimeFeatureExtractor()
        asset_folder = Path(asset_folder)
        
        # Load assets
        asset_files = sorted(list(asset_folder.glob("rt_asset_seed_*.pkl")))
        if not asset_files: raise ValueError(f"No assets found in {asset_folder}")
        print(f"[+] Loading {len(asset_files)} Assets...")
        
        self.boosters = []
        self.scalers_mean = []
        self.scalers_scale = []
        self.all_indices = []
        
        for f in asset_files:
            payload = joblib.load(f)
            self.boosters.append(payload['booster'])
            self.scalers_mean.append(payload['scaler_mean'].astype(np.float32))
            self.scalers_scale.append(payload['scaler_scale'].astype(np.float32))
            idx_list = [self.extractor.name_to_idx[name] for name in payload['features']]
            self.all_indices.append(np.array(idx_list, dtype=np.int32))
        
        self.n_models = len(self.boosters)
        self.scaled_buffers = [np.empty((1, len(idx)), dtype=np.float32, order='C') for idx in self.all_indices]
        self.predictions = np.empty(self.n_models, dtype=np.float32)
        self.best_iterations = [b.best_iteration for b in self.boosters]
        
        # GC Management
        self.frame_counter = 0
        self.gc_interval = 120
        
        print(f"[+] System Ready: RawScore=True | GC=Gen0(100)")

    def predict_frame(self, image: np.ndarray) -> float:
        raw_features = self.extractor.process_frame(image)
        
        for i in range(self.n_models):
            indices = self.all_indices[i]
            buf = self.scaled_buffers[i]
            
            # Fast Copy & Scale
            np.take(raw_features, indices, axis=1, out=buf)
            buf -= self.scalers_mean[i]
            buf /= self.scalers_scale[i]
            
            # Raw Score
            raw_score = self.boosters[i].predict(buf, raw_score=True, num_iteration=self.best_iterations[i])[0]
            
            if raw_score > 0: prob = 1.0 / (1.0 + math.exp(-raw_score))
            else: z = math.exp(raw_score); prob = z / (1.0 + z)
            self.predictions[i] = prob
        
        # Micro-GC: Gentle cleanup to prevent 100MB leaks
        self.frame_counter += 1
        if self.frame_counter >= self.gc_interval:
            gc.collect(0) 
            self.frame_counter = 0

        return float(self.predictions.mean())

def load_engine(engine_path="Engines/Original_RealTimeFrameClassification"):
    return NuclearFastEnsembleClassifier(engine_path)