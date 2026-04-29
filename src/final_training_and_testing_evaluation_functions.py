
################################################################################
# Final Training and Testing/Stressing of Single Models for each operator
################################################################################

import os
import gc
import joblib
import logging
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


def evaluate_stress_level(
    level: int,
    global_dir: str,
    final_model: object,
    model_name: str,
    stress_results_dir: str,
    compression_level: int = 3,
    verbosity: int = 1
):
    """
    Evaluates a trained model on a single level of augmented stress test data.
    """
    # CORRECTED: Path now points to the correctly named scaled stress test file.
    stress_path = os.path.join(
        global_dir, 'stress_test', f"level_{level}",
        f"stress_test_level_{level}_scaled.pkl")

    if not os.path.exists(stress_path):
        raise FileNotFoundError(f"Stress test file not found at: {stress_path}")

    if verbosity:
        logging.info(f"Loading stress test data for level {level} from {stress_path}")

    stress_data_raw = joblib.load(stress_path)
    X_stress = stress_data_raw.drop('label', axis=1, errors='ignore').values
    y_stress = np.array(stress_data_raw['label'])
    del stress_data_raw
    gc.collect()

    stress_preds = final_model.predict_proba(X_stress)[:, 1]
    stress_level_data = {
        "proba_predictions": stress_preds,
        "true_labels": y_stress,
    }

    # CHANGED: Output path is now generic and not tied to an operator.
    stress_level_file = os.path.join(stress_results_dir, f"{model_name}_stress_level_{level}_results.pkl")
    joblib.dump(stress_level_data, stress_level_file, compress=compression_level)
    if verbosity:
        logging.info(f"Stress Test Level {level} results for {model_name} saved in {stress_level_file}")
    
    del X_stress, y_stress, stress_preds
    gc.collect()
    return level, stress_level_data


def final_train_and_evaluate_model(
    global_dir: str,
    best_params: dict,
    model_name: str,
    MODEL_DICT: dict,
    verbosity: int = 1,
    levels: int = None,
    test_results_dir: str = "Results/Final_Results/Test_Results",
    stress_test_results_dir: str = "Results/Final_Results/Stress_Test_Results",
    compression_level: int = 3,
    n_jobs: int = -1
):
     
    """
    Trains a final model on all available data, evaluates on the train and test sets,
    and runs through stress tests. Caches results to avoid re-computation.
    """
    # Define file paths for caching all results
    model_fp = os.path.join(test_results_dir, f"{model_name}_final_model.pkl")
    train_fp = os.path.join(test_results_dir, f"{model_name}_train_results.pkl")
    test_fp = os.path.join(test_results_dir, f"{model_name}_test_results.pkl")

    # 1) Load from cache if all files exist
    if os.path.exists(model_fp) and os.path.exists(train_fp) and os.path.exists(test_fp):
        if verbosity:
            logging.info(f"Loading cached model, train, and test results for {model_name}.")
        final_model = joblib.load(model_fp)
        train_data = joblib.load(train_fp)
        test_data = joblib.load(test_fp)
    else:
        os.makedirs(test_results_dir, exist_ok=True)

        # Load training data
        train_file = os.path.join(global_dir, 'train', "global_train_augmented_and_scaled.pkl")
        train_df = joblib.load(train_file)
        X_train = train_df.drop('label', axis=1, errors='ignore').values
        y_train = np.array(train_df['label'].astype(np.int8))
        del train_df; gc.collect()

        # Train model
        final_model = clone(MODEL_DICT[model_name])
        final_model.set_params(**(best_params or {}), n_jobs=n_jobs) 
        if verbosity:
            logging.info(f"Training final model for {model_name}...")
        final_model.fit(X_train, y_train)
        
        if verbosity:
            logging.info(f"Evaluating model on training data for {model_name}...")
        
        
        train_preds = final_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_preds)
        if verbosity:
            logging.info(f"Train AUC for {model_name}: {train_auc:.4f}")

        train_data = {
            "proba_predictions": train_preds,
            "true_labels": y_train,
        }
        joblib.dump(train_data, train_fp, compress=compression_level)
        if verbosity:
            logging.info(f"Saved train results for {model_name}.")

        del X_train, y_train, train_preds; gc.collect()

        # Load test data
        test_file = os.path.join(global_dir, 'test', "test_scaled.pkl")
        test_df = joblib.load(test_file)
        X_test = test_df.drop('label', axis=1, errors='ignore').values
        y_test = np.array(test_df['label'])
        del test_df; gc.collect()

        # Predict & evaluate test set
        preds = final_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        if verbosity:
            logging.info(f"Test AUC for {model_name}: {auc:.4f}")
        
        # MODIFIED: Removed train_auc from here
        test_data = {
            "proba_predictions": preds, 
            "true_labels": y_test,
        }

        # Save model & test results
        joblib.dump(final_model, model_fp, compress=compression_level)
        joblib.dump(test_data, test_fp, compress=compression_level)
        if verbosity:
            logging.info(f"Saved model & test results for {model_name}.")
        del X_test, y_test, preds; gc.collect()

    # 2) Stress test evaluation (per-level caching)
    stress_data = {}
    if levels:
        os.makedirs(stress_test_results_dir, exist_ok=True)
        for lvl in range(1, levels + 1):
            stress_fp = os.path.join(stress_test_results_dir, f"{model_name}_stress_level_{lvl}_results.pkl")
            if os.path.exists(stress_fp):
                if verbosity:
                    logging.info(f"Loading cached stress results for level {lvl} of {model_name}.")
                stress_data[lvl] = joblib.load(stress_fp)
            else:
                if verbosity:
                    logging.info(f"Evaluating stress level {lvl} for {model_name}...")
                level, data = evaluate_stress_level(
                    level=lvl,
                    global_dir=global_dir,
                    final_model=final_model,
                    model_name=model_name,
                    stress_results_dir=stress_test_results_dir,
                    compression_level=compression_level,
                    verbosity=verbosity
                )
                stress_data[lvl] = data
                
    return final_model, train_data, test_data, stress_data
