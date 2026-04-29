################################################################################
# Stacking ensemble functions
################################################################################

import os
import gc
import re
import joblib
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def train_and_save_stacking_ensemble(
    oof_filepaths: dict,
    stacking_output_folder: str,
    meta_learners: dict,
    candidate_models: list,
    verbosity: int = 1,
    compression_level: int = 3
):
    """
    Trains meta-learners on OOF predictions to create a single best stacking ensemble.
    The best meta-learner is chosen based on the mean of AUC and Accuracy.
    """
    os.makedirs(stacking_output_folder, exist_ok=True)
    history = {}

    def build_meta_dataset(cands):
        # Load the first candidate to determine fold structure
        first_oof_data = joblib.load(oof_filepaths[cands[0]])
        fold_ids = sorted(first_oof_data.keys(), key=int)
        
        Xs, ys = [], []
        for fid in fold_ids:
            feature_columns, y_true = [], None
            for model_name in cands:
                oof_data = joblib.load(oof_filepaths[model_name])
                predictions = np.array(oof_data[fid]["preds"]).reshape(-1, 1)
                feature_columns.append(predictions)
                if y_true is None:
                    y_true = np.array(oof_data[fid]["true_labels"])
            
            Xs.append(np.hstack(feature_columns))
            ys.append(y_true)
            
        return np.vstack(Xs), np.concatenate(ys)

    if verbosity > 0:
        logging.info(f"Building meta-dataset from {len(candidate_models)} candidate models...")
    X_meta, y_meta = build_meta_dataset(candidate_models)

    best_combined_score, best_learner_name = -np.inf, None
    for name, learner in meta_learners.items():
        learner.fit(X_meta, y_meta)
        proba = learner.predict_proba(X_meta)[:, 1]
        preds = learner.predict(X_meta)
        
        # Calculate combined metrics
        auc = roc_auc_score(y_meta, proba)
        acc = accuracy_score(y_meta, preds)
        combined_score = (auc + acc) / 2.0
        
        history[name] = {"auc": auc, "accuracy": acc, "combined_score": combined_score}
        
        if verbosity > 0:
            logging.info(f"Meta-learner '{name}' on OOF data -> AUC: {auc:.4f}, Accuracy: {acc:.4f}, Combined: {combined_score:.4f}")
        
        if combined_score > best_combined_score:
            best_combined_score, best_learner_name = combined_score, name

    if verbosity > 0:
        logging.info(f"Best meta-learner is '{best_learner_name}' with Combined Score: {best_combined_score:.4f}")

    # Retrain the best model on all meta-data and save it
    final_meta_model = meta_learners[best_learner_name]
    final_meta_model.fit(X_meta, y_meta)

    out_path = os.path.join(stacking_output_folder, f"stacking_ensemble_{best_learner_name}.pkl")
    joblib.dump(final_meta_model, out_path, compress=compression_level)
    if verbosity > 0:
        logging.info(f"Saved final stacking model to {out_path}")

    stacking_model_info = {
        "model_path": out_path,
        "candidate_order": candidate_models, # Order of features for the meta-model
    }

    joblib.dump(history, os.path.join(stacking_output_folder, "stacking_meta_history.pkl"), compress=compression_level)
    gc.collect()
    
    return stacking_model_info


def evaluate_stacking_models(
    stacking_model_info: dict,
    test_results_dir: str,
    stress_test_results_dir: str,
    test_output_folder: str,
    stress_output_folder: str,
    compression_level: int = 3,
    verbosity: int = 1
):
    """
    Evaluates the single, overall stacking ensemble on the test and stress sets.
    Logs both AUC and Accuracy and saves both probability and hard predictions.
    """
    def build_meta_dataset_from_results(candidates, results_dir, file_suffix, is_stress=False):
        if not is_stress:
            # For the main test set
            preds, labels = [], None
            for m in candidates:
                res_path = os.path.join(results_dir, f"{m}{file_suffix}")
                res = joblib.load(res_path)
                preds.append(np.array(res["proba_predictions"]).reshape(-1, 1))
                if labels is None: labels = np.array(res["true_labels"])
            return np.hstack(preds), labels
        else:
            # For stress tests, data is grouped by level
            level_data = {}
            pattern = re.compile(r"stress_level_(\d+)")
            for m in candidates:
                for fname in os.listdir(results_dir):
                    if fname.startswith(m) and "stress_level_" in fname:
                        lvl = pattern.search(fname).group(1)
                        res = joblib.load(os.path.join(results_dir, fname))
                        level_data.setdefault(lvl, {"preds": [], "labels": None})
                        level_data[lvl]["preds"].append(np.array(res["proba_predictions"]).reshape(-1, 1))
                        if level_data[lvl]["labels"] is None:
                             level_data[lvl]["labels"] = np.array(res["true_labels"])
            
            meta_datasets = {lvl: (np.hstack(d["preds"]), d["labels"]) for lvl, d in level_data.items()}
            return meta_datasets

    os.makedirs(test_output_folder, exist_ok=True)
    os.makedirs(stress_output_folder, exist_ok=True)
    
    if verbosity: logging.info("Evaluating stacking ensemble...")

    # Load the single stacking model and its candidate order
    model = joblib.load(stacking_model_info["model_path"])
    candidates = stacking_model_info["candidate_order"]
    model_name = os.path.basename(stacking_model_info["model_path"]).replace('.pkl', '')

    # 1. Evaluate on the main test set
    X_test_meta, y_test = build_meta_dataset_from_results(candidates, test_results_dir, "_test_results.pkl")
    proba = model.predict_proba(X_test_meta)[:, 1]
    preds = model.predict(X_test_meta)
    
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test,preds)
    
    if verbosity: 
        logging.info(f"Stacking ensemble test -> AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    
    joblib.dump({"auc": auc, "accuracy": acc,
                 "proba_predictions": proba, "hard_predictions": preds}, 
                os.path.join(test_output_folder, f"{model_name}_test_results.pkl"), 
                compress=compression_level)

    # 2. Evaluate on all stress test levels
    stress_meta_datasets = build_meta_dataset_from_results(candidates, stress_test_results_dir, "", is_stress=True)
    for lvl, (X_s, y_s) in stress_meta_datasets.items():
        proba_s = model.predict_proba(X_s)[:, 1]
        preds_s = model.predict(X_s)

        auc_s = roc_auc_score(y_s, proba_s)
        acc_s = accuracy_score(y_s, preds_s)
        
        if verbosity > 1: 
            logging.info(f"Stacking ensemble stress level {lvl} -> AUC: {auc_s:.4f}, Accuracy: {acc_s:.4f}")
        
        joblib.dump({"auc": auc_s, "accuracy": acc_s,
                     "proba_predictions": proba_s, "hard_predictions": preds_s}, 
                    os.path.join(stress_output_folder, f"{model_name}_stress_level_{lvl}_results.pkl"), 
                    compress=compression_level)
    
    gc.collect()

