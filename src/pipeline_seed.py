################################################################################
# Final pipeline_seed Function (with Stacking Ensemble integration)
################################################################################


import os
import gc
import joblib
import logging
import pandas as pd

from src.bayesian_hyperparameter_search_functions import bayesian_hyperparameter_search
from src.per_seed_directory_creation_and_inner_cross_validaiton_setup import (
    create_directories, create_cv_folds,
    find_and_apply_variance_threshold,
    train_test_split_seed, process_data)
from src.dataset_augmentation_functions import create_random_augmented_dataset
from src.final_training_and_testing_evaluation_functions import final_train_and_evaluate_model
from src.stacking_ensemble_functions import train_and_save_stacking_ensemble, evaluate_stacking_models
from src.voting_ensemble_functions import greedy_ensemble_selection, construct_and_save_voting_ensemble

from config.pipeline_configuration import PipelineConfig


def pipeline_seed(
    gen_results_folder: str,
    seed: int,
    image_df: pd.DataFrame,
    config: PipelineConfig
):
    """
    High-level pipeline that performs:
      1) Directory structure creation.
      2) Global train/test split.
      3) CV fold creation with augmentation and operator processing.
      4) Feature selection per operator (for each model–operator combination).
      5) Bayesian hyperparameter search.
      6) Global processing, stress test creation (cached).
      7) Final training, testing, and stress evaluation (cached).
      8) Voting ensemble evaluation.
      9) Stacking ensemble evaluation.

    Returns a dictionary with pipeline results.
    """
    # Unpack config
    n_splits        = config.n_splits
    TEST_RATIO      = config.test_ratio
    TRAIN_AUG       = config.train_aug_factor
    VAL_AUG         = config.val_aug_factor
    GLOBAL_AUG      = config.global_aug_factor
    BALANCE         = config.balance
    smote           = config.balance_with_smote
    EXTRA_AUG       = config.extra_augmentation_per_class
    N_STRESS        = config.n_stress_test_levels
    verbosity       = config.verbosity
    comp_level      = config.compression_level
    BAYES_OPT       = config.bayesian_opt
    BAYES_STEPS     = config.n_bayesian_steps
    SEARCH_SP       = config.model_hyperparams
    MODEL_DICT      = config.MODEL_DICT
    META_LEARN      = config.meta_models
    VOTE_METHOD     = config.voting_method
    VAR_THRESH      = config.variance_threshold
    TRAIN_OUT_DEL   = config.train_outlier_removal
    CONTAMINATION   = config.contamination
    N_JOBS          = config.n_jobs

    # 1) Directories
    dirs = create_directories(gen_results_folder, seed)
    global_dir    = dirs['global_dir']
    cv_dir        = dirs['cv_splits_dir']
    bayes_hist    = dirs['bayesian_optimization_hystory_dir']
    bayes_oof     = dirs['bayesian_optimization_oof_dir']
    test_dir      = dirs['test_results_dir']
    stress_dir    = dirs['stress_test_results_dir']
    glob_scalers  = dirs['global_scalers_dir']
    loc_scalers   = dirs['local_scalers_dir']

    # 2) Train/Test split
    train_df, test_df = train_test_split_seed(
        image_df, seed, global_dir,
        TEST_RATIO, verbosity, comp_level
    )
    if verbosity:
        logging.info(f"Train size={len(train_df)}, test size={len(test_df)}")

    
    # 3) CV folds (cached inside)
    create_cv_folds(
        train_df, 
        cv_dir, 
        loc_scalers, 
        n_splits, 
        seed, 
        verbosity, 
        comp_level,
        TRAIN_AUG, 
        VAL_AUG, 
        BALANCE,
        smote,
        EXTRA_AUG,
        mask=None, 
        n_jobs=N_JOBS,
        train_outlier_removal=TRAIN_OUT_DEL,
        contamination=CONTAMINATION

    )
    del train_df; gc.collect()

    # 4) Find zero-variance features, save mask, and apply it back to the folds
    features_to_keep_path = find_and_apply_variance_threshold(
        cv_dir=cv_dir,
        verbosity=verbosity,
        compression_level=comp_level,
        threshold = VAR_THRESH
    )

    # 5) Load the generated feature selection mask for subsequent processing
    if os.path.exists(features_to_keep_path):
        if verbosity: logging.info(f"Loading feature selection mask for global processing.")
        feature_mask = joblib.load(features_to_keep_path)
    else:
        if verbosity: logging.warning("Feature selection mask not found. Proceeding with all features.")
        feature_mask = None


    # 5) Bayesian search
    best_params = {}
    for m in MODEL_DICT:
        hist_fp = os.path.join(bayes_hist, f"{m}_bayesian_history.pkl")
        oof_fp  = os.path.join(bayes_oof, f"{m}_oof.pkl")
        if BAYES_OPT and not (os.path.exists(hist_fp) and os.path.exists(oof_fp)):
            if verbosity: logging.info(f"Running Bayes for {m}")
            params = bayesian_hyperparameter_search(
                cv_dir, seed, n_splits,
                SEARCH_SP, BAYES_STEPS,
                hist_fp, oof_fp,
                verbosity, comp_level,
                m, MODEL_DICT,
                n_jobs=config.n_jobs
            )
        else:
            params = {} if not BAYES_OPT else joblib.load(hist_fp)['best_params']
        best_params[(seed,m)] = params

    # 6) Build OOF map
    oof_map = {}
    for fn in os.listdir(bayes_oof):
        if fn.endswith('_oof.pkl'):
            model_name = fn.replace('_oof.pkl', '')
            if model_name in MODEL_DICT:
                oof_map[model_name] = os.path.join(bayes_oof, fn)

    candidates = list(oof_map)

    # 7) Greedy ensemble
    selected, _  = greedy_ensemble_selection(oof_map, candidates, verbosity)

    # 8) Stacking on OOF
    stack_paths = train_and_save_stacking_ensemble(
        oof_filepaths=oof_map,
        stacking_output_folder=os.path.join(bayes_oof, 'stacking_oof_output'),
        meta_learners=META_LEARN,
        candidate_models=candidates,
        verbosity=verbosity,
        compression_level=comp_level
    )

    # ---------------------------------------------------------
    # Final part
    # ---------------------------------------------------------

    # 9) Global data processing (cached)
    logging.info("Starting global data processing...")

    # Define expected output file paths based on the new generic naming convention
    global_train_out = os.path.join(global_dir, 'train', "global_train_augmented_and_scaled.pkl")
    global_test_out = os.path.join(global_dir, 'test', "test_scaled.pkl")
    global_scaler_fp = os.path.join(glob_scalers, "global_scaler.pkl")

    # Check if the final processed files already exist to skip the whole block
    if not (os.path.exists(global_train_out) and os.path.exists(global_test_out) and os.path.exists(global_scaler_fp)):
        # Load original (unprocessed) data
        ori_tr_fp = os.path.join(global_dir, 'train', 'global_training_set.pkl')
        ori_te_fp = os.path.join(global_dir, 'test', 'test_set.pkl')
        
        if not os.path.exists(ori_tr_fp) or not os.path.exists(ori_te_fp):
            raise FileNotFoundError(f"Original global training set ({ori_tr_fp}) or test set ({ori_te_fp}) not found.")
            
        ori_gl_df = joblib.load(ori_tr_fp)
        ori_te_df = joblib.load(ori_te_fp)

        # Augment the data for this specific fold
        if BALANCE is False:
            smote = False

        if smote:
            BALANCE     = False
            EXTRA_AUG   = {}

        # Augment the training data
        aug_gl_df = pd.concat([
            ori_gl_df,
            create_random_augmented_dataset(ori_gl_df, 
                                            GLOBAL_AUG,
                                            balance = BALANCE,
                                            extra_augment_pct = EXTRA_AUG)], 
                                            ignore_index=True)


        # Process the global training and test sets using the refactored function
        status = process_data(
            seed = seed,
            train_fold=aug_gl_df,
            val_fold=ori_te_df,
            mask=feature_mask,
            dir=global_dir,
            fold_idx=None, 
            compression_level=comp_level,
            mod='global',
            scal_dir=glob_scalers,
            level=None,
            smote=smote,
            train_outlier_removal=TRAIN_OUT_DEL,
            contamination=CONTAMINATION

        )
        if verbosity: logging.info(status)
    else:
        if verbosity: logging.info("[Global] Skipping data processing as final files already exist.")


    # 10) Stress test processing (cached per level)
    logging.info("Starting stress test processing...")

    # We still need the original test data for augmentation
    ori_te_fp = os.path.join(global_dir, 'test', 'test_set.pkl')
    if not os.path.exists(ori_te_fp):
        raise FileNotFoundError(f"Original test set ({ori_te_fp}) not found for stress testing.")
    ori_te_df = joblib.load(ori_te_fp)

    for lvl in range(1, N_STRESS + 1):
        stress_level_dir = os.path.join(global_dir, 'stress_test', f'level_{lvl}')
        os.makedirs(stress_level_dir, exist_ok=True)
        
        # Create augmented data for this specific stress level
        df_stress = create_random_augmented_dataset(ori_te_df, n_augmentations=lvl)

        # Process each stress level. The `process_data` function handles the
        # internal file existence check for this specific level.
        status = process_data(
            seed = seed,
            train_fold=pd.DataFrame(), # No training data for stress test processing
            val_fold=df_stress,
            mask=feature_mask,
            dir=stress_level_dir, # Output directory is specific to the level
            fold_idx=None,
            compression_level=comp_level,
            mod='stress',
            scal_dir=glob_scalers, # Directory where the global_scaler.pkl is located
            level=lvl,
            smote = False
        )
        if verbosity: logging.info(status)

    # 11) Final train/test/stress per model (cached)
    final_models, final_tests, final_stress = {}, {}, {}
    # ADDED: A new dictionary to store the training results
    final_train_results = {}

    for m in MODEL_DICT:
        logging.info(f"Starting final training and evaluation for model: {m}")
        model_best_params = best_params.get((seed, m), {})

        fm, train_res, test_res, stress_res = final_train_and_evaluate_model(
            global_dir=global_dir,
            best_params=model_best_params,
            model_name=m,
            MODEL_DICT=MODEL_DICT,
            verbosity=verbosity,
            levels=N_STRESS,
            test_results_dir=test_dir,
            stress_test_results_dir=stress_dir,
            compression_level=comp_level,
            n_jobs=config.n_jobs
        )
        
        # Store all the results in their respective dictionaries
        final_models[(seed, m)] = fm
        final_train_results[(seed, m)] = train_res
        final_tests[(seed, m)] = test_res
        final_stress[(seed, m)] = stress_res
        
    # 12) Voting ensemble evaluation
    vot_out = os.path.join(test_dir, 'voting_ensemble_results')
    st_vot_out = os.path.join(stress_dir, 'voting_ensemble_results')
    os.makedirs(vot_out, exist_ok=True)
    os.makedirs(st_vot_out, exist_ok=True)
    construct_and_save_voting_ensemble(
        test_dir, stress_dir, vot_out, st_vot_out,
        VOTE_METHOD, selected, comp_level, verbosity
    )

    # 13) Stacking evaluation
    st_test_dir = os.path.join(test_dir,'stacking_ensemble_results')
    st_str_dir  = os.path.join(stress_dir,'stacking_ensemble_results')
    os.makedirs(st_test_dir, exist_ok=True)
    os.makedirs(st_str_dir, exist_ok=True)
    evaluate_stacking_models(
        stack_paths, test_dir, stress_dir,
        st_test_dir, st_str_dir, comp_level, verbosity
    )

    return {
        'best_params': best_params,
        'selected_ensemble': selected,
        'stack_paths': stack_paths,
        'final_models': final_models,
    }



