from pathlib import Path

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex




# ==============================================================================
# Global Settings
# ==============================================================================

SEEDS               = range(10)
RESULTS_ROOT        = Path('Dataset++')
IMAGES_DIR          = RESULTS_ROOT / Path('Articles_Images')
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DF_PATH       = RESULTS_ROOT / 'image_df.pkl'

# ==============================================================================
# GLOBAL PLOTTING STYLE (Nature Journal Standard)
# ==============================================================================
def set_nature_style(font_family='serif', font_serif=['Times New Roman'], font_size=12):
    """Configures Matplotlib/Seaborn to follow Nature Journal standards[cite: 2, 3]."""
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set_context("paper") 
    sns.set_style("white")
    
    plt.rcParams['font.family']      = font_family 
    plt.rcParams['font.serif']       = font_serif
    plt.rcParams['font.size']        = font_size 
    plt.rcParams['axes.labelsize']   = font_size + 2 
    plt.rcParams['xtick.labelsize']  = font_size 
    plt.rcParams['ytick.labelsize']  = font_size 
    plt.rcParams['legend.fontsize']  = font_size 
    plt.rcParams['figure.titlesize'] = font_size + 4 
    
    plt.rcParams['axes.linewidth']      = 1.0 
    plt.rcParams['axes.edgecolor']      = 'black' 
    plt.rcParams['axes.spines.top']     = False 
    plt.rcParams['axes.spines.right']   = False 
    plt.rcParams['axes.grid']           = False 
    
    plt.rcParams['xtick.direction']     = 'out' 
    plt.rcParams['ytick.direction']     = 'out' 
    plt.rcParams['xtick.major.size']    = 3 
    plt.rcParams['ytick.major.size']    = 3 
    
    plt.rcParams['xtick.minor.visible'] = True 
    plt.rcParams['ytick.minor.visible'] = False 


# ==============================================================================
# COLOR PALETTES
# ==============================================================================
_raw_cb = sns.color_palette("colorblind", 10) 
_hex_cb = [to_hex(rgb) for rgb in _raw_cb] 
COLOR_BLIND_PALETTE = {
    'blue': _hex_cb[0], 'orange': _hex_cb[1], 'green': _hex_cb[2],
    'red': _hex_cb[3], 'purple': _hex_cb[4], 'brown': _hex_cb[5],
    'pink': _hex_cb[6], 'grey': _hex_cb[7], 'amber': _hex_cb[8], 
    'light_blue': _hex_cb[9]
}

CLASS_COLORS = {
    1: '#0072B2', '1': '#0072B2', 'Beta': '#0072B2',
    0: '#E69F00', '0': '#E69F00', 'Alpha': '#E69F00' 
}

# --- B. MODEL COLORS (All Aliases Included) ---
MODEL_COLORS = {
    # Tree-based (Greens)
    'RandomForest': COLOR_BLIND_PALETTE['orange'], 'Random Forest': COLOR_BLIND_PALETTE['orange'], 'RF': COLOR_BLIND_PALETTE['orange'],
    'ExtraTrees': COLOR_BLIND_PALETTE['amber'], 'Extra Trees': COLOR_BLIND_PALETTE['amber'], 'ET':COLOR_BLIND_PALETTE['amber'],
    
    # Boosting (Oranges)
    'XGBoost': COLOR_BLIND_PALETTE['pink'], 'XGB': COLOR_BLIND_PALETTE['pink'],
    'LGBM': COLOR_BLIND_PALETTE['light_blue'], 'LightGBM': COLOR_BLIND_PALETTE['light_blue'],
    
    # Linear/Distance (Blues)
    'KNeighbors': '#000000', 'KNN': '#000000',
    'LogisticRegression': COLOR_BLIND_PALETTE['blue'], 'Logistic Regression': COLOR_BLIND_PALETTE['blue'], 'LR': COLOR_BLIND_PALETTE['blue'],
    
    # Ensembles (Purples/Pinks/Greys)
    'stacking_ensemble_KNeighbors': COLOR_BLIND_PALETTE['purple'], 'Stacking KNN': COLOR_BLIND_PALETTE['purple'], 'Stack(KNN)': COLOR_BLIND_PALETTE['purple'],
    'stacking_ensemble_GradientBoosting': COLOR_BLIND_PALETTE['purple'], 'Stacking GBM': COLOR_BLIND_PALETTE['purple'], 'Stack(GBM)': COLOR_BLIND_PALETTE['purple'],
    'stacking_ensemble_Logistic': COLOR_BLIND_PALETTE['purple'], 'Stacking Logistic': COLOR_BLIND_PALETTE['purple'], 'Stack(LR)': COLOR_BLIND_PALETTE['purple'],
    'best_subset_soft_voting': COLOR_BLIND_PALETTE['grey'], 'Greedy Voting': COLOR_BLIND_PALETTE['grey'], 'Greedy': COLOR_BLIND_PALETTE['grey'],
    'all_models_soft_voting':'#2E8B57', 'All Models Voting':'#2E8B57', 'Voting': '#2E8B57'
}




# ==============================================================================
# MAPPINGS MODELS and FEATURE NAMES
# ==============================================================================
VALID_MODEL_NAMES = [
    'RandomForest', # --> RF
    'ExtraTrees',   # --> ET
    'LGBM',         # --> LGBM
    'XGBoost',      # --> XGB
    'KNeighbors',   # --> KNN
    'LogisticRegression',  # --> LR
    'stacking_ensemble_KNeighbors', # --> Stack_KNN
    'stacking_ensemble_GradientBoosting',   # -->  Stack_LGBM
    'stacking_ensemble_Logistic', # --> S_LR
    'best_subset_soft_voting', # --> Greed_Voting
    'all_models_soft_voting'   # --> All_Votnig
]

NAME_MAPPING_LONG = {
    'stacking_ensemble_GradientBoosting': 'Stacking GBM',
    'stacking_ensemble_KNeighbors':       'Stacking KNN',
    'stacking_ensemble_Logistic':         'Stacking Logistic',
    'best_subset_soft_voting':            'Greedy Voting',
    'all_models_soft_voting':             'All Models Voting',
    'LogisticRegression':                 'Logistic Regression',
    'KNeighbors':                         'KNN',
    'RandomForest':                       'Random Forest',
    'ExtraTrees':                         'Extra Trees',
    'LGBM':                               'LightGBM',
    'XGBoost':                            'XGBoost'
}

NAME_MAPPING_SHORT = {
    'All Models Voting': 'Voting', 'Greedy Voting': 'Greedy',
    'Stacking KNN': 'Stack(KNN)', 'Stacking GBM': 'Stack(GBM)', 'Stacking Logistic': 'Stack(LR)',
    'KNN': 'KNN', 'Random Forest': 'RF', 'Extra Trees': 'ET',
    'XGBoost': 'XGB', 'LightGBM': 'LGBM', 'Logistic Regression': 'LR'
}

tech_shape = [ 
    'shp_area', 'shp_perimeter', 'shp_aspect_ratio', 'shp_extent',
    'shp_solidity', 'shp_circularity', 'shp_eq_diameter', 'shp_eccentricity',
    'shp_vertices', 'shp_major_axis', 'shp_minor_axis', 'orientation', 'inertia_eigval1', 'inertia_eigval2',
    'bbox_aspect_ratio', 'caliper_diameter_proxy', 'extreme_width', 'extreme_height', 'shp_convexity',
    'shp_num_defects', 'shp_mean_defect_depth', 'shp_max_defect_depth', 
]

pub_shape = [ 
    'Area', 'Perimeter', 'Aspect Ratio', 'Extent',
    'Solidity', 'Circularity', 'Eq. Diameter', 'Eccentricity',
    'Vertices', 'Major Axis', 'Minor Axis', 'Orientation', 'Inertia Eigval 1', 'Inertia Eigval 2',
    'BBox Aspect Ratio', 'Caliper Diameter', 'Extreme Width', 'Extreme Height', 'Convexity',
    'Num Defects', 'Mean Defect Depth', 'Max Defect Depth', 
]


LTP_P = 8
FFT_DESCRIPTOR = 11

tech_ltp_u = [f'ltp_u_{i}' for i in range(LTP_P+2)]
tech_ltp_l = [f'ltp_l_{i}' for i in range(LTP_P+2)]
tech_hu    = [f'hu_{i}' for i in range(7)]
tech_fft   = [f'fft_{i}' for i in range(FFT_DESCRIPTOR)]
tech_shape = [ 
    'shp_area', 'shp_perimeter', 'shp_aspect_ratio', 'shp_extent',
    'shp_solidity', 'shp_circularity', 'shp_eq_diameter', 'shp_eccentricity',
    'shp_vertices', 'shp_major_axis', 'shp_minor_axis', 'orientation', 'inertia_eigval1', 'inertia_eigval2',
    'bbox_aspect_ratio', 'caliper_diameter_proxy', 'extreme_width', 'extreme_height', 'shp_convexity',
    'shp_num_defects', 'shp_mean_defect_depth', 'shp_max_defect_depth', 
]

technical_names_master = tech_ltp_u + tech_ltp_l + tech_hu + tech_fft + tech_shape

# --- B. Define Master Publication List (Order MUST match Technical) ---
pub_ltp_u = [f'LTP Upper Pat {i}' for i in range(LTP_P+2)]
pub_ltp_l = [f'LTP Lower Pat {i}' for i in range(LTP_P+2)]
pub_hu    = [f'Hu Moment {i}' for i in range(7)]
pub_fft   = [f'FFT Coef {i}' for i in range(FFT_DESCRIPTOR)]
pub_shape = [ 
    'Area', 'Perimeter', 'Aspect Ratio', 'Extent',
    'Solidity', 'Circularity', 'Eq. Diameter', 'Eccentricity',
    'Vertices', 'Major Axis', 'Minor Axis', 'Orientation', 'Inertia Eigval 1', 'Inertia Eigval 2',
    'BBox Aspect Ratio', 'Caliper Diameter', 'Extreme Width', 'Extreme Height', 'Convexity',
    'Num Defects', 'Mean Defect Depth', 'Max Defect Depth', 
]

publication_names_master    = pub_ltp_u + pub_ltp_l + pub_hu + pub_fft + pub_shape
TECH_TO_PUB                 = dict(zip(technical_names_master, publication_names_master))

# ==============================================================================
# Defining Resolution of Images
# ==============================================================================
IMAGE_DPI           = 600


# ==============================================================================
# Performance Metrics to Analyze and Precedent BenchMarks
# ==============================================================================
ANALYSIS_METRICS = ['AUC', 'accuracy', 'precision', 'recall', 'f1', 'pr_auc']
REF_VALUES       = {
    'precision': {'alpha': 0.75, 'beta': 0.94},
    'recall':    {'alpha': 0.81, 'beta': 0.91},
    'f1':        {'alpha': 0.78, 'beta': 0.92},
    'auc':       {'alpha': 0.86, 'beta': 0.86}
}