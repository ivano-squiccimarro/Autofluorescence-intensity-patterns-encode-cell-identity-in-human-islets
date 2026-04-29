import os
import gc
import joblib
import logging

import numpy as np
import pandas as pd
from skopt import gp_minimize
from sklearn.base import clone
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, accuracy_score


def eval_fold(fold, param_dict, model_name, model_dict):
    """
    Train & evaluate a single fold. Returns (true_labels, preds, auc, acc).
    """
    X_tr, y_tr, X_vl, y_vl = fold

    # Clone, set hyperparams, fit, and predict
    clf = clone(model_dict[model_name])
    clf.set_params(**param_dict)
    clf.fit(X_tr, y_tr)
    preds = clf.predict_proba(X_vl)[:, 1]
    acc_preds = clf.predict(X_vl)

    # Calculate metrics
    auc = roc_auc_score(y_vl, preds)
    acc = accuracy_score(y_vl,acc_preds)

    # Clean up
    del clf
    return y_vl, preds, auc, acc


def bayesian_hyperparameter_search(
    cv_dir: str,
    seed: int,
    n_splits: int,
    search_space: dict,
    n_calls: int = 40,
    bayesian_history_path: str = None,
    bayesian_oof_path: str = None,
    verbosity: int = 1,
    compression_level: int = 3,
    model_name: str = "RandomForest",
    model_dict: dict = None,
    n_jobs: int = 1
):
    """
    Performs Bayesian hyperparameter optimization for a given model.
    """
    if model_dict is None:
        raise ValueError("model_dict must be provided.")

    # Extract the search space for this model
    try:
        space_list = list(search_space[model_name].values())
        param_names = list(search_space[model_name].keys())
    except KeyError:
        logging.warning(f"No search space found for model '{model_name}'. Skipping.")
        return {}
    except Exception:
        raise ValueError("search_space must be a dict of skopt dimensions per model_name.")

    # Load train/val folds from disk
    fold_data = []
    for fold_id in range(1, n_splits + 1):
        base = os.path.join(cv_dir, f"Fold_{fold_id}")
        # NOTE: Ensure these file paths are correct for your new pipeline
        tr_path = os.path.join(base, f"train_fold_{fold_id}_augmented_and_scaled.pkl")
        vl_path = os.path.join(base, f"val_fold_{fold_id}_augmented_and_scaled.pkl")

        if not os.path.exists(tr_path) or not os.path.exists(vl_path):
            raise FileNotFoundError(f"Could not find pre-processed fold data at {tr_path} or {vl_path}")

        tr = joblib.load(tr_path)
        vl = joblib.load(vl_path)

        X_tr = tr.drop("label", axis=1).values
        y_tr = tr["label"].values # Keep as NumPy array

        X_vl = vl.drop("label", axis=1).values
        y_vl = vl["label"].values

        fold_data.append((X_tr, y_tr, X_vl, y_vl))

    # Storage for per-iteration results
    per_iter_fold_results = []
    best_combined_score = -np.inf
    best_oof_cache = None

    def objective(params):
        """
        The objective function for the Bayesian optimizer.
        Now optimizes for the mean of AUC and Accuracy.
        """
        nonlocal best_combined_score, best_oof_cache

        param_dict = dict(zip(param_names, params))

        # --- NEW, MORE ROBUST FIX ---
        for key, value in param_dict.items():
            if isinstance(value, np.generic):
                param_dict[key] = value.item()

        # --- END OF FIX ---
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_fold)(fold, param_dict, model_name, model_dict)
            for fold in fold_data
        )
        aucs = [r[2] for r in results]
        accs = [r[3] for r in results]
        mean_auc = np.mean(aucs)
        mean_acc = np.mean(accs)

        # New combined score objective
        combined_score = (mean_auc + mean_acc) / 2.0

        per_iter_fold_results.append(results)
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_oof_cache = results

        return -combined_score

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space_list,
        n_calls=n_calls,
        random_state=seed,
        verbose=(verbosity > 0)
    )

    best_params = dict(zip(param_names, result.x))
    # CHANGED: Updated logging to reflect the new combined score.
    logging.info(f"Model '{model_name}': best hyperparams => {best_params} with combined score: {-result.fun:.4f}")

    # Build a DataFrame of the optimization history
    df_history = pd.DataFrame(result.x_iters, columns=param_names)
    
    # Extract detailed metrics for history
    mean_aucs = [np.mean([fold[2] for fold in iter_res]) for iter_res in per_iter_fold_results]
    mean_accs = [np.mean([fold[3] for fold in iter_res]) for iter_res in per_iter_fold_results]

    df_history["mean_auc"] = mean_aucs
    df_history["mean_accuracy"] = mean_accs
    df_history["combined_score"] = -result.func_vals
    df_history["fold_results"] = per_iter_fold_results

    # Save full history
    opt_history = {
        "x_iters": result.x_iters,
        "func_vals": result.func_vals,
        "best_params": best_params,
        "space": space_list,
        "param_names": param_names,
        "fold_results_per_iter": per_iter_fold_results,
        "df_history": df_history,
    }
    joblib.dump(opt_history, bayesian_history_path, compress=compression_level)
    logging.info(f"Optimization history saved to {bayesian_history_path}")

    # Save out-of-fold predictions
    oof = {
        fold_id: {"true_labels": res[0], "preds": res[1]}
        for fold_id, res in enumerate(best_oof_cache, start=1)
    }
    joblib.dump(oof, bayesian_oof_path, compress=compression_level)
    if verbosity > 1:
        logging.info(f"Saved OOF predictions to {bayesian_oof_path}")

    # Cleanup
    del oof
    gc.collect()

    return best_params
