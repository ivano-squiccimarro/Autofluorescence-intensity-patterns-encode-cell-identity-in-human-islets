#  --------------- --------------- --------------- --------------- ---------------
# Creating Directories and Prefixed train/test and Train/Validation 
#  --------------- --------------- --------------- --------------- ---------------

import os
import joblib
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.feature_extraction_functions import extract_feats
from src.dataset_augmentation_functions import create_random_augmented_dataset

def create_directories(
        general_result_folder:str,
        seed: int) -> dict:
    """
    Create a consistent directory structure for a given seed.

    Structure:

    seed_dir/
    ├── Datasets/
    │    ├── Training_Test_Splits/
    │    │      ├── Global_Data/
    │    │      │      ├── train/
    │    │      │      ├── test/
    │    │      │      ├── stress_test/
    │    │      │      └── Global_Scalers/
    │    │      └── CV_splits/
    │    │             ├── train/
    │    │             ├── validation/
    │    │             └── Local_Scalers/
    └── Results/
         ├── Bayesian_Optimization_Results/
         │     └── bayesian_optimization_hystory/
         │     └── oof_predictions/
         └── Final_Results/
                ├── Test_Results/
                └── Stress_Test_Results/

    Returns
    -------
    dirs : dict
        A dictionary with keys:
          - seed_dir: Root directory for the given seed.
          - dataset_dir: Path to Datasets/Training_Test_Splits.
          - global_dir: Path to Global_Data.
          - global_train_dir: Path to Global_Data/train.
          - global_test_dir: Path to Global_Data/test.
          - global_stress_test_dir: Path to Global_Data/stress_test.
          - global_scalers_dir: Path to Global_Data/Global_Scalers.
          - cv_splits_dir: Path to CV_splits.
          - local_scalers_dir: Path to CV_splits/Local_Scalers.
          - results_dir: Root directory for Results.
          - bayesian_optimization_results_dir: Path to Results/Bayesian_Optimization_Results.
          - bayesian_optimization_hystory_dir: Path to Results/Bayesian_Optimization_Results/bayesian_optimization_hystory.
          - bayesian_optimization_oof_dir: Path to Results/Bayesian_Optimization_Results/oof_predictions.
          - final_results_dir: Path to Results/Final_Results.
          - test_results_dir: Path to Results/Final_Results/Test_Results.
          - stress_test_results_dir: Path to Results/Final_Results/Stress_Test_Results.
  """


    # 1) Seed-level root folder
    if general_result_folder is None:
        seed_dir = f"Results_{seed}"
    else:
        seed_dir = f"{general_result_folder}/Results_{seed}"
    os.makedirs(seed_dir, exist_ok=True)

    # 2) Datasets branch
    dataset_dir = os.path.join(seed_dir, "Datasets", "Training_Test_Splits")
    os.makedirs(dataset_dir, exist_ok=True)

    # Global Data subfolders
    global_dir = os.path.join(dataset_dir, "Global_Data")
    os.makedirs(global_dir, exist_ok=True)

    global_train_dir = os.path.join(global_dir, "train")
    os.makedirs(global_train_dir, exist_ok=True)

    global_test_dir = os.path.join(global_dir, "test")
    os.makedirs(global_test_dir, exist_ok=True)

    global_stress_test_dir = os.path.join(global_dir, "stress_test")
    os.makedirs(global_stress_test_dir, exist_ok=True)

    global_scalers_dir = os.path.join(global_dir, "Global_Scalers")
    os.makedirs(global_scalers_dir, exist_ok=True)

    # CV Splits subfolders
    cv_splits_dir = os.path.join(dataset_dir, "CV_splits")
    os.makedirs(cv_splits_dir, exist_ok=True)

    local_scalers_dir = os.path.join(cv_splits_dir, "Local_Scalers")
    os.makedirs(local_scalers_dir, exist_ok=True)

    # 3) Results branch
    results_dir = os.path.join(seed_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)

    # 3A) Bayesian Optimization (without plots)
    bayesian_optimization_results_dir = os.path.join(results_dir, "Bayesian_Optimization_Results")
    os.makedirs(bayesian_optimization_results_dir, exist_ok=True)

    bayesian_optimization_hystory_dir = os.path.join(bayesian_optimization_results_dir, "bayesian_optimization_history")
    os.makedirs(bayesian_optimization_hystory_dir, exist_ok=True)

    bayesian_optimization_oof_dir = os.path.join(bayesian_optimization_results_dir, "oof_predictions")
    os.makedirs(bayesian_optimization_oof_dir, exist_ok=True)

    # 3C) Final Model Performances
    final_results_dir = os.path.join(results_dir, "Final_Results")
    os.makedirs(final_results_dir, exist_ok=True)

    test_results_dir = os.path.join(final_results_dir, "Test_Results")
    os.makedirs(test_results_dir, exist_ok=True)

    stress_test_results_dir = os.path.join(final_results_dir, "Stress_Test_Results")
    os.makedirs(stress_test_results_dir, exist_ok=True)

    # Return dictionary with all directories
    dirs = {
        "seed_dir": seed_dir,
        "dataset_dir": dataset_dir,
        "global_dir": global_dir,
        "global_train_dir": global_train_dir,
        "global_test_dir": global_test_dir,
        "global_stress_test_dir": global_stress_test_dir,
        "global_scalers_dir": global_scalers_dir,
        "cv_splits_dir": cv_splits_dir,
        "local_scalers_dir": local_scalers_dir,
        "results_dir": results_dir,
        "bayesian_optimization_oof_dir": bayesian_optimization_oof_dir,
        "bayesian_optimization_hystory_dir": bayesian_optimization_hystory_dir,
        "final_results_dir": final_results_dir,
        "test_results_dir": test_results_dir,
        "stress_test_results_dir": stress_test_results_dir
    }

    return dirs

def train_test_split_seed(
    image_df: pd.DataFrame,
    seed: int,
    global_dir: str,
    TEST_RATIO: int = 0.2,
    verbosity: int = 1,
    compression_level:int = 3
)-> pd.DataFrame:

    """
    Perform a train/test split and save the global training and test sets.
    """

    if verbosity > 0:
        logging.info(f"Creating train/test split for seed {seed}...")


    training_file = os.path.join(global_dir, 'train', 'global_training_set.pkl')
    test_file = os.path.join(global_dir, 'test', 'test_set.pkl')

    if os.path.exists(training_file) and os.path.exists(test_file):
        if verbosity > 0:
          logging.info("Train/test split files found. Loading them.")
        ori_global_train_df = joblib.load(training_file)
        test_df = joblib.load(test_file)
    else:
        if verbosity > 0:
          logging.info(f"Train/test split files not found. Creating new splits. TEST_RATIO : {TEST_RATIO}..... seed: {seed}")
        ori_global_train_df, test_df = train_test_split(
            image_df,
            test_size=TEST_RATIO,
            random_state=seed,
            stratify=image_df['label'] if 'label' in image_df.columns else None
        )
        joblib.dump(ori_global_train_df, training_file, compress = compression_level)
        joblib.dump(test_df, test_file, compress = compression_level)

        if verbosity > 0:
          logging.info(f"Train/test split saved:\n  - {training_file}\n  - {test_file}")
    return ori_global_train_df, test_df

def apply_outlier_removal(
    df: pd.DataFrame, 
    mask: Optional[list], 
    contamination: float = 0.01, 
    seed: int = 42
) -> pd.DataFrame:
    """
    Extracts features from the dataframe, applies IsolationForest, 
    and returns a filtered dataframe without outliers.
    """
    if len(df) < 50:
        # Too few samples to reliably detect outliers
        return df

    # 1. Extract Features to detect outliers on
    # We use extract_feats helper you already have
    feats_df = extract_feats(df)
    
    # 2. Apply Mask if provided (to focus outlier detection on relevant features)
    if mask is not None:
        features_to_keep = [col for col in mask if col in feats_df.columns]
        if features_to_keep:
            feats_df = feats_df[features_to_keep]

    # Remove label if present in features (we want unsupervised detection)
    if 'label' in feats_df.columns:
        feats_df = feats_df.drop('label', axis=1)

    X = feats_df.values
    
    # 3. Scale Features (Crucial for distance/variance based outlier detection)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Detect
    # n_jobs=1 ensures no parallel conflicts if called inside Parallel loops
    iso = IsolationForest(contamination=contamination, random_state=seed, n_jobs=1)
    preds = iso.fit_predict(X_scaled) # -1 = Outlier, 1 = Inlier

    # 5. Filter
    inlier_mask = preds == 1
    df_clean = df.iloc[inlier_mask].copy()
    
    n_removed = len(df) - len(df_clean)
    # Ideally logging should be handled by caller to avoid spam in parallel, 
    # but we can just return the clean DF.
    
    return df_clean

################################################################################
# Inner Cross_validation creation and Processing Functions
################################################################################


def transform_df_pair(
        seed:int,
        df_train: pd.DataFrame,
        df_valtest: pd.DataFrame, 
        mod: str,
        mask: list = None, 
        scaler: StandardScaler = None,
        smote: bool = False,
        verbosity: int = 1
):
    """
    It processes a pair of DataFrames by extracting features, 
    scaling correctly, applying a name-based mask, and optionally applying smote.
    """
    
    # --- Logic for 'inner_cv' and 'global' (Training Mode) ---
    if mod in ['inner_cv', 'global']:
        # 1. Feature Extraction
        if verbosity > 0:
            logging.info(f"Extracting features from training DataFrame with {len(df_train)} rows.") 
        train_features_df = extract_feats(df_train)
        
        if verbosity > 0:
            logging.info(f"Extracting features from validation/test DataFrame with {len(df_valtest)} rows.")
        valtest_features_df = extract_feats(df_valtest)
 
        if mask is not None:
            features_to_keep = [col for col in mask if col in train_features_df.columns]
            columns_to_keep = features_to_keep + ['label'] 
            train_features_df = train_features_df[columns_to_keep]
            valtest_features_df = valtest_features_df[columns_to_keep]
  
        # Get the final list of feature names (after masking)
        feature_names = train_features_df.drop('label', axis=1).columns.tolist()

        # Separate features (X) and labels (Y)
        X_train = train_features_df.drop('label', axis=1).values
        Y_train = train_features_df['label'].values
        X_valtest = valtest_features_df.drop('label', axis=1).values
        Y_valtest = valtest_features_df['label'].values 
        
        
        # 2. Scaling
        if scaler is None:
            if verbosity > 0: 
                logging.info("No scaler provided. Fitting a new one on Train set.")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else: 
            if verbosity > 0: 
                logging.info("Scaler provided.")
            X_train_scaled = scaler.transform(X_train)
            
        X_valtest_scaled = scaler.transform(X_valtest)

        # 4. Conditional smote Application
        if smote:
            oversampler = SMOTE(random_state=seed)
            X_train_balanced, Y_train_balanced = oversampler.fit_resample(
                X_train_scaled, 
                Y_train
            )
            X_train_final = X_train_balanced
            Y_train_final = Y_train_balanced
        else:
            X_train_final = X_train_scaled
            Y_train_final = Y_train

        # 5. Re-assign the processed features back to the DataFrames
        df_train_final = pd.DataFrame(
            data=X_train_final,
            columns=feature_names)
        df_train_final['label'] = Y_train_final
        
        df_valtest_final = pd.DataFrame(
            data=X_valtest_scaled,
            columns=feature_names )
        df_valtest_final['label'] = Y_valtest # Use the Y_valtest variable
        
        return df_train_final, df_valtest_final, scaler

    # --- Logic for 'stress' (Testing Mode) ---
    elif mod == 'stress':
        if verbosity > 0:
            logging.info(f"Extracting features from stress test DataFrame with {len(df_valtest)} rows.")
        valtest_features_df = extract_feats(df_valtest)
        

        if mask is not None:
            features_to_keep = [col for col in mask if col in valtest_features_df.columns]
            columns_to_keep = features_to_keep + ['label']
            valtest_features_df = valtest_features_df[columns_to_keep]

        # Get the final list of feature names (after masking)
        feature_names = valtest_features_df.drop('label', axis=1).columns.tolist() 
        
        X_valtest = valtest_features_df.drop('label', axis=1).values
        Y_valtest = valtest_features_df['label'].values 

        X_valtest_scaled = scaler.transform(X_valtest)

        df_valtest_final = pd.DataFrame(
            data=X_valtest_scaled,
            columns=feature_names 
        )
        df_valtest_final['label'] = Y_valtest 
        return df_valtest_final
        
    return None

def process_data(
    seed: int,
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    mask: Optional[np.ndarray],
    dir: str,
    fold_idx: Optional[int],
    compression_level: int,
    mod: str,
    scal_dir: str,
    level: Optional[int] = None,
    smote: bool = False,
    train_outlier_removal: bool = False,
    contamination: float = 0.01
) -> str:
    """
    This function is takes a train and validation/test set and process then 
    at the same time with the same conditions.
    """
    local_train = train_fold.copy(deep=True)
    local_val = val_fold.copy(deep=True)

    # === INNER CV ===
    if mod == 'inner_cv' and fold_idx is not None:
        fold_dir = os.path.join(dir, f"Fold_{fold_idx}")
        scal_fold_dir = os.path.join(scal_dir, f"Fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(scal_fold_dir, exist_ok=True)

        train_out = os.path.join(fold_dir, f"train_fold_{fold_idx}_augmented_and_scaled.pkl")
        val_out = os.path.join(fold_dir, f"val_fold_{fold_idx}_augmented_and_scaled.pkl")
        scaler_fp = os.path.join(scal_fold_dir, f"scaler_fold_{fold_idx}.pkl")

        if os.path.exists(train_out) and os.path.exists(val_out) and os.path.exists(scaler_fp):
            return f"[Inner CV] Fold {fold_idx} already processed."

        scaler = None 
        op_tr, op_val, op_scaler = transform_df_pair(
            seed=seed,
            df_train=local_train, 
            df_valtest=local_val,
            mask=mask, 
            scaler=scaler, 
            mod=mod,
            smote=smote
        )

        joblib.dump(op_tr, train_out, compress=compression_level)
        joblib.dump(op_val, val_out, compress=compression_level)
        joblib.dump(op_scaler, scaler_fp, compress=compression_level)

        return f"[Inner CV] Fold {fold_idx} processed."

    # === GLOBAL ===
    if mod == 'global':
        train_out = os.path.join(dir, 'train', "global_train_augmented_and_scaled.pkl")
        test_out = os.path.join(dir, 'test', "test_scaled.pkl")
        scaler_fp = os.path.join(scal_dir, "global_scaler.pkl")

        if os.path.exists(train_out) and os.path.exists(test_out) and os.path.exists(scaler_fp):
            return "[Global] Data already processed."
        
        if train_outlier_removal:
            logging.info(f"[Global] Applying outlier removal with contamination={contamination}...")
            local_train = apply_outlier_removal(
                local_train, 
                mask=mask, 
                contamination=contamination, 
                seed=seed
            )
        
        # For the global run, we fit a new scaler on the full (augmented) training set
        scaler = None

        op_tr, op_val, op_scaler = transform_df_pair(
            seed=seed,
            df_train=local_train, 
            df_valtest=local_val,
            mask=mask, 
            scaler=scaler, 
            mod=mod,
            smote=smote
        )

        joblib.dump(op_tr, train_out, compress=compression_level)
        joblib.dump(op_val, test_out, compress=compression_level)
        joblib.dump(op_scaler, scaler_fp, compress=compression_level)

        return "[Global] Train/Test sets processed."

    # === STRESS ===
    if mod == 'stress':
        stress_out = os.path.join(dir, f"stress_test_level_{level}_scaled.pkl")
        scaler_fp = os.path.join(scal_dir, "global_scaler.pkl")

        if os.path.exists(stress_out):
            return f"[Stress] Level {level} already processed."
        
        # Stress tests MUST use the globally fitted scaler
        scaler = joblib.load(scaler_fp)
        empty_df = pd.DataFrame(columns=val_fold.columns) # No training data for stress test

        op_val = transform_df_pair(
            seed=seed,
            df_train=empty_df, 
            df_valtest=local_val,
            mask=mask, 
            scaler=scaler, 
            mod=mod,
            smote=smote
        )

        joblib.dump(op_val, stress_out, compress=compression_level)
        return f"[Stress] Level {level} processed."

    return f"[Error] Unrecognized mode '{mod}'."

def _process_single_fold(
    seed: int,
    fold_data: pd.DataFrame, 
    global_train_df: pd.DataFrame, 
    cv_dir: str, 
    local_scaler_dir: str,
    mask: list, 
    TRAIN_AUG_FACTOR: int, 
    VAL_AUG_FACTOR: int, 
    BALANCE: bool,
    smote:bool,
    EXTRA_AUG: dict,
    compression_level: int,
    train_outlier_removal: bool = False,
    contamination: float = 0.01
):
    """Helper function to process one CV fold. Target for parallel execution."""
    fold_idx, (train_idx, val_idx) = fold_data
    
    train_fold = global_train_df.iloc[train_idx].copy()
    val_fold = global_train_df.iloc[val_idx].copy()

    # Eliminate Statistical Outliers from Training Dataset

    if train_outlier_removal:

        train_fold = apply_outlier_removal(
            train_fold, 
            mask=mask, 
            contamination=contamination, 
            seed=seed
        )


    # Augment the data for this specific fold
    if BALANCE is False:
        smote = False

    if smote:
        BALANCE     = False
        EXTRA_AUG   = {}

    if TRAIN_AUG_FACTOR > 0:
        train_fold = pd.concat([train_fold, 
                                create_random_augmented_dataset(train_fold, 
                                                                n_augmentations=TRAIN_AUG_FACTOR,
                                                                balance = BALANCE,
                                                                extra_augment_pct = EXTRA_AUG)], 
                                                                ignore_index=True)
    if VAL_AUG_FACTOR > 0:
        val_fold = pd.concat([val_fold, 
                            create_random_augmented_dataset(val_fold, 
                                                            n_augmentations=VAL_AUG_FACTOR,
                                                            balance = BALANCE,
                                                            extra_augment_pct = EXTRA_AUG)], 
                                                            ignore_index=True)
        
    return process_data(
        seed = seed,
        train_fold=train_fold,
        val_fold=val_fold,
        mask=mask,
        dir=cv_dir,
        fold_idx=fold_idx + 1,
        compression_level=compression_level,
        mod='inner_cv',
        scal_dir=local_scaler_dir,
        smote = smote
    )

def create_cv_folds(
    global_train_df: pd.DataFrame,
    cv_dir: str,
    local_scaler_dir: str,
    n_splits: int,
    seed: int,
    verbosity: int,
    compression_level: int,
    TRAIN_AUG_FACTOR: int,
    VAL_AUG_FACTOR: int,
    BALANCE: bool,
    smote: bool,
    EXTRA_AUG: dict,
    mask: Optional[np.ndarray],
    n_jobs: int,
    train_outlier_removal: bool = False,
    contamination:float = 0.01
    ) -> None:
    """
    Creates and processes all cross-validation folds in parallel.
    """
    os.makedirs(cv_dir, exist_ok=True)

    y = global_train_df['label'].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(enumerate(skf.split(global_train_df, y)))

    if verbosity > 0:
        logging.info(f"Starting parallel processing of {n_splits} CV folds (n_jobs={n_jobs})...")

    # This is the parallelized fold creation loop you requested
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_fold)(
            seed,
            fold_data,
            global_train_df,
            cv_dir,
            local_scaler_dir,
            mask,
            TRAIN_AUG_FACTOR,
            VAL_AUG_FACTOR,
            BALANCE,
            smote,
            EXTRA_AUG,
            compression_level,
            train_outlier_removal,
            contamination
        ) for fold_data in tqdm(folds, desc="Processing CV Folds")
    )

    if verbosity > 0:
        logging.info("Parallel processing of CV folds complete.")
        for result in results:
            logging.info(result)

def find_and_apply_variance_threshold(
    cv_dir: str, 
    verbosity: int, 
    compression_level: int, 
    threshold: float = 0.0,
    non_feature_cols: list = 'label'
) -> str:
    """
    Analyzes processed CV folds to find and remove features with consistently
    low variance, preserving original Feature Generation Order.
    """
    if verbosity > 0:
        logging.info("Identifying features with low variance across all CV folds...")

    selector = VarianceThreshold(threshold=threshold)
    all_folds_kept_features, feature_count = None, 0
    fold_dirs = sorted([d for d in os.listdir(cv_dir) if d.startswith('Fold_') and os.path.isdir(os.path.join(cv_dir, d))])

    # --- Part 1: Find the consistent feature mask ---
    for fold_name in tqdm(fold_dirs, desc="Scanning Folds for Variance"):
        train_path = os.path.join(cv_dir, fold_name, f'train_{fold_name.lower()}_augmented_and_scaled.pkl')
        if not os.path.exists(train_path):
            continue
        
        train_df = joblib.load(train_path)
        
        # --- FIX: Preserve Order ---
        all_columns_ordered = train_df.columns.tolist()
        feature_cols = [col for col in all_columns_ordered if col not in non_feature_cols]

        feature_count = len(feature_cols)
        if not feature_cols:
            continue
            
        # Extract using the ORDERED list
        X_train = train_df[feature_cols].values
        
        if X_train.ndim != 2 or X_train.shape[0] == 0:
            continue
            
        selector.fit(X_train)
        
        # Get the boolean mask of features to keep
        support_mask = selector.get_support()
        
        # Apply mask to the ORDERED list of names
        current_fold_kept_features = set([feature_cols[i] for i, kept in enumerate(support_mask) if kept])
        
        if all_folds_kept_features is None: 
            all_folds_kept_features = current_fold_kept_features
        else: 
            all_folds_kept_features.intersection_update(current_fold_kept_features)

    if all_folds_kept_features is None:
        logging.error("Could not determine features to keep. No valid folds found."); return ""
    
    final_kept_feature_names = [col for col in feature_cols if col in all_folds_kept_features]

    dataset_dir = os.path.dirname(cv_dir)
    features_to_keep_path = os.path.join(dataset_dir, 'features_to_keep.pkl')
    joblib.dump(final_kept_feature_names, features_to_keep_path, compress=compression_level)
    
    if verbosity > 0:
        removed = feature_count - len(final_kept_feature_names)
        logging.info(f"Removed {removed} features with variance <= {threshold} consistently across folds.")
        logging.info(f"Feature mask saved to: {features_to_keep_path}")

    # --- Part 2: Apply the mask back to all folds ---
    if verbosity > 0: logging.info("Applying variance threshold mask back to all CV folds...")
    
    # Kept features are now in Logical Generation Order
    kept_features = final_kept_feature_names

    for fold_name in tqdm(fold_dirs, desc="Applying Mask to Folds"):
        for data_type in ['train', 'val']:
            file_name = f'{data_type}_{fold_name.lower()}_augmented_and_scaled.pkl'
            path = os.path.join(cv_dir, fold_name, file_name)
            
            if os.path.exists(path):
                df = joblib.load(path)
                if 'label' not in df.columns: continue
                
                y_df = df.pop('label')
                
                # Re-select columns ensuring order matches kept_features
                # Check for existence first to be safe
                existing_cols = set(df.columns)
                cols_to_keep_features = [col for col in kept_features if col in existing_cols]
                
                df = df[cols_to_keep_features]
                df['label'] = y_df
                
                joblib.dump(df, path, compress=compression_level)
    
    if verbosity > 0: logging.info("Finished applying mask to all folds.")
    return features_to_keep_path