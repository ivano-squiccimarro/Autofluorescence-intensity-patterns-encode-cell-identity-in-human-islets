import joblib
import numpy         as np
from pathlib         import Path
from sklearn.metrics import ( average_precision_score, roc_auc_score)

from config.presentation_config import VALID_MODEL_NAMES


# --- HELPER AND METRIC FUNCTIONS (Unchanged) ---
def compute_all_metrics(y_true, proba):
    if not hasattr(y_true, '__len__') or not hasattr(proba, '__len__') or y_true is None or proba is None: return {}
    if len(y_true) == 0 or len(proba) == 0: return {}
    preds = (proba >= 0.5).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    metrics = {}
    metrics['precision_1'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall_1'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['f1_1'] = 2 * metrics['precision_1'] * metrics['recall_1'] / (metrics['precision_1'] + metrics['recall_1']) if (metrics['precision_1'] + metrics['recall_1']) > 0 else 0.0
    metrics['precision_0'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['recall_0'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['f1_0'] = 2 * metrics['precision_0'] * metrics['recall_0'] / (metrics['precision_0'] + metrics['recall_0']) if (metrics['precision_0'] + metrics['recall_0']) > 0 else 0.0
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics['mcc'] = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
    has_both_classes = len(np.unique(y_true)) > 1
    metrics['AUC'] = roc_auc_score(y_true, proba) if has_both_classes else np.nan
    metrics['pr_auc'] = average_precision_score(y_true, proba) if has_both_classes else np.nan
    metrics['precision'] = (metrics['precision_0'] + metrics['precision_1']) / 2
    metrics['recall'] = (metrics['recall_0'] + metrics['recall_1']) / 2
    metrics['f1'] = (metrics['f1_0'] + metrics['f1_1']) / 2
    return metrics

def get_labels_from_data(data_obj):
    if data_obj and data_obj.get('true_labels') is not None: return np.array(data_obj['true_labels'])
    if data_obj and data_obj.get('true_label') is not None: return np.array(data_obj['true_label'])
    return None

# --- REFACTORED DATA LOADING FUNCTION ---
def process_seed_optimized(name_folder, seed, multiple_sources=False):
    """
    Refactored to explicitly search for files based on a predefined list of valid model names,
    and to correctly handle missing labels in ensemble result files.
    """
    records = []
    final_results_path = Path(f"{name_folder}/Results_{seed}/Results/Final_Results")
    base_path = final_results_path / "Test_Results"
    voting_path = base_path / "voting_ensemble_results"
    stacking_path = base_path / "stacking_ensemble_results"
    stress_path = final_results_path / "Stress_Test_Results"

    # --- Step 1: Find Reference Labels ---
    ref_train_labels, ref_test_labels = None, None
    try:
        ref_train_data = joblib.load(base_path / "RandomForest_train_results.pkl")
        ref_train_labels = get_labels_from_data(ref_train_data)
        ref_test_data = joblib.load(base_path / "RandomForest_test_results.pkl")
        ref_test_labels = get_labels_from_data(ref_test_data)
        if ref_test_labels is None:
             print(f"CRITICAL WARNING for Seed {seed}: Reference test labels could not be loaded from RandomForest file.")
    except FileNotFoundError:
        print(f"CRITICAL WARNING for Seed {seed}: Could not load reference labels from RandomForest files.")


    # --- Step 2: Process Train and Test data for all models ---
    for model_name in VALID_MODEL_NAMES:
        is_base = model_name in ['RandomForest', 'ExtraTrees', 'LGBM', 'XGBoost', 'KNeighbors', 'LogisticRegression']
        is_voting = "voting" in model_name
        is_stacking = "stacking" in model_name

        try:
            # Training Data (only for base models)
            if is_base and ref_train_labels is not None:
                train_file = base_path / f"{model_name}_train_results.pkl"
                if train_file.is_file():
                    data = joblib.load(train_file)
                    metrics = compute_all_metrics(ref_train_labels, data.get('proba_predictions'))
                    if metrics:
                        if multiple_sources:
                            records.append({"seed": seed, "data_split": "Training", 
                                            "model_type": "base", "model_name": model_name, 
                                            "stress_level": -1, "data_source": name_folder,
                                            **metrics})
                        else:
                            records.append({"seed": seed, "data_split": "Training", 
                                            "model_type": "base", "model_name": model_name, 
                                            "stress_level": -1,**metrics})

            # Test Data (for all model types)
            test_data, model_type = None, None
            if is_base:
                test_file = base_path / f"{model_name}_test_results.pkl"
                if test_file.is_file(): test_data, model_type = joblib.load(test_file), "base"
            elif is_voting:
                potentials = list(voting_path.glob(f"{model_name}*.pkl"))
                if potentials: test_data, model_type = joblib.load(potentials[0]), "voting"
            elif is_stacking:
                potentials = list(stacking_path.glob(f"{model_name}*.pkl"))
                if potentials: test_data, model_type = joblib.load(potentials[0]), "stacking"

            if test_data and ref_test_labels is not None:
                proba = test_data.get('proba_predictions', test_data.get('predictions'))
                # Use the single source of truth (ref_test_labels) for all models on the test set.
                metrics = compute_all_metrics(ref_test_labels, proba)
                if metrics:
                    if multiple_sources:
                        records.append({"seed": seed, "data_split": "Test", "model_type": model_type, 
                                        "model_name": model_name, "stress_level": 0, "data_source": name_folder,
                                        **metrics})
                    else:
                        records.append({"seed": seed, "data_split": "Test", 
                                            "model_type": "base", "model_name": model_name, 
                                            "stress_level": 0,**metrics})

                    
            elif test_data and ref_test_labels is None:
                print(f"Warning for Seed {seed}, Model {model_name} (Test): Skipping due to missing reference test labels.")

        except Exception as e:
            print(f"ERROR for Seed {seed}, Model {model_name} (Train/Test): {e}")

    # --- Step 3: Process Stress Data separately ---
    if stress_path.is_dir():
        for level in range(1, 4):
            ref_stress_labels = None
            try:
                ref_stress_file = stress_path / f"RandomForest_stress_level_{level}_results.pkl"
                if ref_stress_file.is_file():
                    ref_stress_data = joblib.load(ref_stress_file)
                    ref_stress_labels = get_labels_from_data(ref_stress_data)
                if ref_stress_labels is None:
                    print(f"Warning for Seed {seed}, Stress Level {level}: Could not load reference labels from RandomForest file. Skipping this level.")
                    continue 
            except Exception as e:
                print(f"ERROR loading reference labels for Seed {seed}, Stress Level {level}: {e}")
                continue

            # Now loop through all models for this specific stress level
            for model_name in VALID_MODEL_NAMES:
                is_base = model_name in ['RandomForest', 'ExtraTrees', 'LGBM', 'XGBoost', 'KNeighbors', 'LogisticRegression']
                is_voting = "voting" in model_name
                is_stacking = "stacking" in model_name

                try:
                    stress_file, mtype = None, None
                    if is_base:
                        path_to_check = stress_path / f"{model_name}_stress_level_{level}_results.pkl"
                        if path_to_check.is_file(): stress_file, mtype = path_to_check, "base"
                    elif is_stacking:
                        path_to_check = stress_path / "stacking_ensemble_results" / f"{model_name}_stress_level_{level}_results.pkl"
                        if path_to_check.is_file(): stress_file, mtype = path_to_check, "stacking"
                    elif is_voting:
                        path_to_check = stress_path / "voting_ensemble_results" / f"{model_name}_stress_level_{level}.pkl"
                        if path_to_check.is_file(): stress_file, mtype = path_to_check, "voting"

                    if stress_file:
                        data = joblib.load(stress_file)
                        proba = data.get('proba_predictions', data.get('predictions'))

                        # Use the reference labels loaded for this level
                        metrics = compute_all_metrics(ref_stress_labels, proba)

                        if metrics:
                            records.append({"seed": seed, "data_split": "Stress", "model_type": mtype,
                                            "model_name": model_name, "stress_level": level, "data_source": name_folder,
                                            **metrics})
                except Exception as e:
                    print(f"ERROR for Seed {seed}, Model {model_name} (Stress Level {level}): {e}")

    return records

