################################################################################
# Voting Ensemble functions
################################################################################


import os
import re
import gc
import joblib
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def greedy_ensemble_selection(
    oof_filepaths: dict,
    candidate_models: list,
    verbose: int = 1
):
    """
    Greedy forward selection of an ensemble subset based on a combined score (AUC + Accuracy).
    Loads OOF predictions lazily to limit RAM.
    Returns a tuple (selected_models, history) where history is a list of (models_list, score).
    """
    history = []
    selected = []
    best_score = 0.0
    candidates = candidate_models.copy()

    def load_oof_data(models):
        # Load OOF dicts and get fold IDs
        oof_data = {}
        fold_ids = None
        for m in models:
            data = joblib.load(oof_filepaths[m])
            oof_data[m] = data
            if fold_ids is None:
                fold_ids = sorted(data.keys(), key=lambda x: int(x))
        return oof_data, fold_ids

    def compute_ensemble_score(models):
        oof_data, fold_ids = load_oof_data(models)
        scores = []
        for fid in fold_ids:
            agg = None
            for m in models:
                preds = np.array(oof_data[m][fid]["preds"])
                agg = preds.copy() if agg is None else agg + preds
            agg /= len(models)
            true = np.array(oof_data[models[0]][fid]["true_labels"])
            
            # Calculate both metrics for the fold
            auc = roc_auc_score(true, agg)
            class_preds = (agg >= 0.5).astype(int)
            acc = accuracy_score(true, class_preds)
            
            # Combined score for the fold
            scores.append((auc + acc) / 2.0)
            
        gc.collect()
        return float(np.mean(scores))

    # 1) Evaluate single-model candidates
    if verbose > 0:
        logging.info("Initial candidate evaluation:")
    best_candidate = None
    for m in candidates:
        score = compute_ensemble_score([m])
        if verbose > 0:
            logging.info(f"Candidate {m} Combined Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_candidate = m

    if best_candidate is None:
        raise ValueError("No candidate yielded a score improvement.")

    selected.append(best_candidate)
    history.append((selected.copy(), best_score))
    if verbose > 0:
        logging.info(f"Starting ensemble with: {best_candidate} (Score: {best_score:.4f})")
    candidates.remove(best_candidate)
    current_score = best_score

    # 2) Greedy forward addition
    while candidates:
        best_add = None
        for m in candidates:
            trial = selected + [m]
            trial_score = compute_ensemble_score(trial)
            if verbose > 0:
                logging.info(f"Trial {trial} Score: {trial_score:.4f}")
            if trial_score > current_score:
                current_score = trial_score
                best_add = m
        if best_add is None:
            if verbose > 0:
                logging.info("No further improvement; stopping selection.")
            break
        selected.append(best_add)
        history.append((selected.copy(), current_score))
        candidates.remove(best_add)
        if verbose > 0:
            logging.info(f"Added {best_add}, new ensemble Score: {current_score:.4f}")

    return selected, history



def construct_and_save_voting_ensemble(
    test_results_dir: str,
    stress_test_results_dir: str,
    test_output_folder: str,
    stress_output_folder: str,
    voting_method: str = "soft",
    selected_subset: list = None,
    compression_level: int = 3,
    verbosity: int = 1
):
    """
    Computes and saves voting ensembles, evaluating based on a combined score (AUC + Accuracy).
    Saves both probability and hard predictions for analysis.
    """
    os.makedirs(test_output_folder, exist_ok=True)
    os.makedirs(stress_output_folder, exist_ok=True)

    def _compute_ensemble(probs_array, method):
        if method == "soft": # Average probabilities
            return np.mean(probs_array, axis=1)
        else: # Average hard predictions (majority vote)
            return np.mean((probs_array >= 0.5).astype(int), axis=1)

    # --- Load Test Set Results ---
    if verbosity: logging.info("Loading test results for voting ensemble...")
    model_test_probs = {}
    test_labels = None
    test_files = [f for f in os.listdir(test_results_dir) if f.endswith("_test_results.pkl")]
    for fname in test_files:
        model_name = fname.replace("_test_results.pkl", "")
        res = joblib.load(os.path.join(test_results_dir, fname))
        model_test_probs[model_name] = np.array(res["proba_predictions"])
        if test_labels is None:
            test_labels = np.array(res["true_labels"])

    # 1. Overall ensemble (all models)
    all_models = list(model_test_probs.keys())
    if all_models:
        non_logistics_models = [k for k in all_models if "LogisticRegression" not in k]
        probs_arr = np.column_stack([model_test_probs[k] for k in non_logistics_models])
        ensemble_probs = _compute_ensemble(probs_arr, voting_method)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)

        auc = roc_auc_score(test_labels, ensemble_probs)
        acc = accuracy_score(test_labels, ensemble_preds)
        score = (auc + acc) / 2.0

        if verbosity: logging.info(f"Overall voting ensemble ({len(all_models)} models) -> AUC: {auc:.4f}, Acc: {acc:.4f}")
        joblib.dump({"auc": auc, "accuracy": acc, "combined_score": score, 
                     "proba_predictions": ensemble_probs, "hard_predictions": ensemble_preds, "keys": all_models},
                    os.path.join(test_output_folder, f"all_models_{voting_method}_voting_ensemble_results.pkl"),
                    compress=compression_level)

    # 2. Best subset ensemble (from greedy selection)
    if selected_subset:
        valid_subset = [k for k in selected_subset if k in model_test_probs]
        if valid_subset:
            probs_arr = np.column_stack([model_test_probs[k] for k in valid_subset])
            sub_probs = _compute_ensemble(probs_arr, voting_method)
            sub_preds = (sub_probs >= 0.5).astype(int)
            
            auc = roc_auc_score(test_labels, sub_probs)
            acc = accuracy_score(test_labels, sub_preds)
            score = (auc + acc) / 2.0
            
            if verbosity: logging.info(f"Best-subset voting ensemble ({len(valid_subset)} models) -> AUC: {auc:.4f}, Acc: {acc:.4f}")
            joblib.dump({"auc": auc, "accuracy": acc, "combined_score": score,
                         "proba_predictions": sub_probs, "hard_predictions": sub_preds, "keys": valid_subset},
                        os.path.join(test_output_folder, f"best_subset_{voting_method}_voting_ensemble_results.pkl"),
                        compress=compression_level)

    # --- Load Stress Test Results ---
    if verbosity: logging.info("Loading stress test results for voting ensemble...")
    stress_files = [f for f in os.listdir(stress_test_results_dir) if "stress_level_" in f]
    lvl_pat = re.compile(r"stress_level_(\d+)")
    model_stress_probs = {} # model_name -> {level: preds_array}
    stress_labels = {} # level -> labels_array

    for fname in stress_files:
        match = lvl_pat.search(fname)
        if not match: continue
        lvl = match.group(1)
        model_name = fname.split(f"_stress_level_{lvl}")[0]
        res = joblib.load(os.path.join(stress_test_results_dir, fname))
        model_stress_probs.setdefault(model_name, {})[lvl] = np.array(res["proba_predictions"])
        if lvl not in stress_labels:
            stress_labels[lvl] = np.array(res["true_labels"])

    # 3. Evaluate ensembles on each stress level
    for lvl in sorted(stress_labels.keys()):
        models_for_lvl = [m for m, d in model_stress_probs.items() if lvl in d]
        if not models_for_lvl: continue

        # 3a) Overall ensemble on stress level
        all_arr = np.column_stack([model_stress_probs[k][lvl] for k in models_for_lvl])
        overall_probs = _compute_ensemble(all_arr, voting_method)
        overall_preds = (overall_probs >= 0.5).astype(int)
        
        auc = roc_auc_score(stress_labels[lvl], overall_probs)
        acc = accuracy_score(stress_labels[lvl], overall_preds)
        
        
        if verbosity > 1: logging.info(f"Stress Lvl {lvl}: Overall voting -> AUC: {auc:.4f}, Acc: {acc:.4f}")
        joblib.dump({"auc": auc, "accuracy": acc, 
                     "proba_predictions": overall_probs, "hard_predictions": overall_preds, "keys": models_for_lvl},
                    os.path.join(stress_output_folder, f"all_models_{voting_method}_voting_stress_level_{lvl}.pkl"),
                    compress=compression_level)

        # 3b) Best-subset ensemble on stress level
        if selected_subset:
            valid_subset = [k for k in selected_subset if lvl in model_stress_probs.get(k, {})]
            if valid_subset:
                sub_arr = np.column_stack([model_stress_probs[k][lvl] for k in valid_subset])
                sub_probs = _compute_ensemble(sub_arr, voting_method)
                sub_preds = (sub_probs >= 0.5).astype(int)

                auc = roc_auc_score(stress_labels[lvl], sub_probs)
                acc = accuracy_score(stress_labels[lvl], sub_preds)
    
                
                if verbosity > 1: logging.info(f"Stress Lvl {lvl}: Best-subset voting -> AUC: {auc:.4f}, Acc: {acc:.4f}")
                joblib.dump({"auc": auc, "accuracy": acc,
                             "proba_predictions": sub_probs, "hard_predictions": sub_preds, "keys": valid_subset},
                            os.path.join(stress_output_folder, f"best_subset_{voting_method}_voting_stress_level_{lvl}.pkl"),
                            compress=compression_level)
    gc.collect()

