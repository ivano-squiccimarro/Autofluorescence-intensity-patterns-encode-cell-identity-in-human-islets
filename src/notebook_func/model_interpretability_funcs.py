import os
import shap
import math
import joblib
from   joblib               import Parallel, delayed

from   pathlib              import Path

import pandas               as pd
import numpy                as np
import seaborn              as sns

import matplotlib.pyplot     as plt
import matplotlib.image      as mpimg
from   matplotlib.colors     import LinearSegmentedColormap
from   matplotlib.ticker     import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm.auto               import tqdm

from sklearn.decomposition   import PCA
from sklearn.inspection      import PartialDependenceDisplay, permutation_importance

from config.presentation_config import (
    RESULTS_ROOT,
    IMAGE_DPI,
    COLOR_BLIND_PALETTE,
    set_nature_style
)





# ========================================================================
# Permutation Features Importance and Partial Dedependence Plot Functions
# ========================================================================

# ________________________________________________________________________
# Helpers

def load_data_for_seed(folder, seed, data_type='test', feature_names=None):
    """
    Loads data and RESTORES column names using the explicitly passed feature_names.
    """
    import joblib
    import numpy as np
    import pandas as pd
    
    try:
        path_segment = "test/test_scaled.pkl" if data_type == 'test' else "train/global_train_augmented_and_scaled.pkl"
        data_path = folder / f"Results_{seed}/Datasets/Training_Test_Splits/Global_Data/{path_segment}"

        df = joblib.load(data_path)
        X = df.drop(columns=['label'])
        y = np.array(df['label'].tolist())

        # FIX: Check if columns are missing names, and map them using the provided list
        if isinstance(X.columns, pd.RangeIndex) or (len(X.columns) > 0 and isinstance(X.columns[0], int)):
            if feature_names is not None and len(X.columns) == len(feature_names):
                X.columns = feature_names
            else:
                print(f"⚠️ Warning: Column mismatch in seed {seed}. Data={len(X.columns)}, Names={len(feature_names) if feature_names else 'None provided'}")
        
        return X, y
    except Exception as e:
        print(f"ERROR: Could not load {data_type} set for seed {seed}. Error: {e}")
        return None, None
    

def load_data_raw(seed, model_name):
    """
    Loads data and model WITHOUT renaming columns.
    Returns X with columns 0, 1, 2...
    """
    try:
        # 1. Load Data
        data_path = RESULTS_ROOT / f"Results_{seed}/Datasets/Training_Test_Splits/Global_Data/test/test_scaled.pkl"
        if not data_path.exists(): return None, None
        
        df = joblib.load(data_path)
        
        # Robust Drop
        if 'label' in df.columns:
            X = df.drop(columns=['label'])
        else:
            X = df.iloc[:, :-1]
            
        # 2. Load Model
        model_path = RESULTS_ROOT / f"Results_{seed}/Results/Final_Results/Test_Results/{model_name}_final_model.pkl"
        if not model_path.exists(): return None, None
        
        model = joblib.load(model_path)
        return model, X

    except Exception as e:
        print(f"Load Error (Seed {seed}): {e}")
        return None, None
    
def plot_feature_importance(
    importance_df, 
    model_name, 
    raw_data_df=None, 
    title_suffix="", 
    add_numbers=False, 
    save_path=None,
    top_n=20,
    show_=True,
    figsize=(12, 10)
):
    """
    Creates a horizontal bar plot for feature importances with error bars.
    Added: save_path argument to save the figure.
    """
    importance_df = importance_df.sort_values(by='mean_importance', ascending=True).tail(top_n)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figsize)
    xerr_data = importance_df['std_importance'] if 'std_importance' in importance_df.columns else None
    
    ax = sns.barplot(x='mean_importance', y=importance_df.index, data=importance_df, color='skyblue', xerr=xerr_data)
    ax.set_title(f'Feature Importance for {model_name}\n{title_suffix}', fontsize=18, weight='bold')
    ax.set_xlabel('Mean Importance Score', fontsize=12)
    ax.set_ylabel('Feature Name', fontsize=12)
    
    plt.tight_layout()
    
    # --- Save Logic ---
    if save_path:
        plt.savefig(save_path, dpi=IMAGE_DPI, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    if show_:
        plt.show()
    else:
        plt.close()

def load_model_importance(seed, model_name, feature_names):
    """Worker function to load one model and extract its importances."""
    path_to_check = RESULTS_ROOT / f"Results_{seed}/Results/Final_Results/Test_Results/{model_name}_final_model.pkl"
    if path_to_check.is_file():
        try:
            model = joblib.load(path_to_check)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return {'seed': seed, 'model_name': model_name, **dict(zip(feature_names, importances))}
        except Exception as e:
            print(f"Warning: Could not process model {path_to_check}. Error: {e}")
    return None

# ________________________________________________________________________
# Gini and Permutation Feature Importance

def get_gini_importance(model_name, seeds, tech_names, save_dir, tech_to_pub_map,
                        show_=True, save_ = False, figsize = (8,8)):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = save_dir / f"{model_name}_Gini_Summary.csv"
    raw_path     = save_dir / f"{model_name}_Gini_Raw_Seeds.csv"

    if summary_path.exists() and raw_path.exists():
        print(f"--- 🟢 Loading Cached Gini Importance ---")
        summary_df = pd.read_csv(summary_path, index_col=0)
        raw_df     = pd.read_csv(raw_path, index_col=0)
    else:
        print(f"--- 🟡 Extracting Gini Importance (Technical Order) ---")
        # 1. Extract using TECHNICAL names (this restores the order 0..N)
        tasks = [(seed, model_name) for seed in seeds]
        results = Parallel(n_jobs=-1)(
            delayed(load_model_importance)(seed, m_name, tech_names) for seed, m_name in tasks
        )
        raw_df = pd.DataFrame([res for res in results if res is not None])
        
        # 2. Normalize
        feat_cols = [c for c in raw_df.columns if c not in ['seed', 'model_name']]
        raw_df[feat_cols] = raw_df[feat_cols].div(raw_df[feat_cols].sum(axis=1) + 1e-9, axis=0)

        # 3. Create Summary
        summary_df = pd.DataFrame({
            'mean_importance': raw_df[feat_cols].mean(axis=0),
            'std_importance': raw_df[feat_cols].std(axis=0)
        })
        
        # 4. CRITICAL: Map to Publication Names HERE, just before saving/plotting
        summary_df.index = [tech_to_pub_map.get(x, x) for x in summary_df.index]
        raw_df.columns = [tech_to_pub_map.get(x, x) if x in feat_cols else x for x in raw_df.columns]

        summary_df.to_csv(summary_path)
        raw_df.to_csv(raw_path)

    plot_feature_importance(
        importance_df=summary_df,
        model_name=model_name,
        raw_data_df=raw_df,
        title_suffix="(Gini Impurity)",
        save_path=save_dir / f"{model_name}_Gini_Plot.png" if save_ else None,
        top_n=10, show_=show_, figsize=figsize
    )
    return summary_df, raw_df

def get_top_n_importance_df(feature_names, model_name, seeds, load_data_fn, results_root, top_n=5, n_repeats=10):
    """
    Calculates permutation importance across seeds and returns the summary DataFrame
    for the top N features.
    """
    print(f"\n--- ANALYSIS: Permutation Importance (Aggregating {len(seeds)} seeds) ---")

    all_perm_importances = []
    
    # Iterate through seeds to gather importance data
    for seed in tqdm(seeds, desc=f"Permutation Imp for {model_name}"):
        # User must provide their specific data loader function here
        X_test_df, y_test = load_data_fn(seed, data_type='test')
        if X_test_df is None: continue

        try:
            # Construct path (Assumes standard structure, adjust if needed)
            model_path = results_root / f"Results_{seed}/Results/Final_Results/Test_Results/{model_name}_final_model.pkl"
            model = joblib.load(model_path)
            result = permutation_importance(
                model, X_test_df.values, y_test, 
                n_repeats=n_repeats, random_state=seed, n_jobs=-1, scoring='roc_auc'
            )
            all_perm_importances.append(result.importances_mean)
        except Exception as e:
            print(f"Skipping seed {seed}: {e}")

    if not all_perm_importances:
        print("No results found.")
        return None

    # Aggregate results
    perm_matrix = np.array(all_perm_importances)
    perm_importance_df = pd.DataFrame({
        'mean_importance': perm_matrix.mean(axis=0),
        'std_importance': perm_matrix.std(axis=0)
    }, index=feature_names)
    
    # Sort and take top N
    top_n_df = perm_importance_df.sort_values(by='mean_importance', ascending=False).head(top_n)
    
    return top_n_df


def analyze_permutation_importance_across_seeds(
    tech_names, 
    tech_to_pub_map, 
    model_name, 
    folder_name, 
    seeds, 
    top_n=5, 
    n_repeats=10, 
    save_dir=None,
    save_plot=True,
    show_=True
):
    """Calculates permutation importance for available seeds and aggregates."""
    print(f"\n--- Permutation Importance Aggregation (seeds: {list(seeds)}) ---")

    all_perm_importances = []
    for seed in tqdm(seeds, desc="Processing Seeds"):
        X_test, y_test = load_data_for_seed(folder_name, seed, data_type='test', feature_names=tech_names)
        model_path = folder_name / f"Results_{seed}/Results/Final_Results/Test_Results/{model_name}_final_model.pkl"
        
        if X_test is None or not model_path.exists():
            continue

        try:
            model = joblib.load(model_path)
            result = permutation_importance(model, X_test.values, y_test, n_repeats=n_repeats, random_state=seed, n_jobs=-1, scoring='roc_auc')
            all_perm_importances.append(result.importances_mean)
        except Exception as e:
            print(f"Error Seed {seed}: {e}")

    if not all_perm_importances:
        print("❌ No valid seed results found for Permutation Importance.")
        return []

    # Aggregate
    perm_matrix = np.array(all_perm_importances)
    pub_names = [tech_to_pub_map.get(name, name) for name in tech_names]
    
    perm_df = pd.DataFrame({
        'mean_importance': perm_matrix.mean(axis=0),
        'std_importance': perm_matrix.std(axis=0)
    }, index=pub_names)
    
    top_df = perm_df.sort_values(by='mean_importance', ascending=False).head(top_n)

    # --- ADDED: Save CSVs for Decoupled Plotting ---
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Summary CSV
        top_df.to_csv(save_dir / f"{model_name}_Permutation_Summary_Top{top_n}.csv")
        
        # Save Raw Seeds CSV (filtered to top_n features to save space)
        raw_df = pd.DataFrame(perm_matrix, columns=pub_names)
        raw_df_top = raw_df[top_df.index]
        raw_df_top.index.name = 'seed'
        raw_df_top.to_csv(save_dir / f"{model_name}_Permutation_Raw_Seeds_Top{top_n}.csv")

    if save_plot or show_:
        save_p = save_dir / f"{model_name}_Permutation_Importance_Top{top_n}.png" if (save_dir and save_plot) else None
        plot_feature_importance(top_df, model_name, title_suffix="(Permutation)", save_path=save_p, show_=show_, top_n=top_n)

    return top_df



# ________________________________________________________________________
# Partial Dependence Plots

def _get_pdp_data(model, X_test, feature):
    """
    Calculates PDP values without drawing the plot to screen.
    Returns (average_pdp, grid_values).
    """
    display = PartialDependenceDisplay.from_estimator(
        model, 
        X_test, 
        features=[feature], 
        kind='average', 
        grid_resolution=75,
        ax=None 
    )
    pdp_values = display.pd_results[0]['average'][0]
    grid_values = display.axes_[0, 0].lines[0].get_xdata()
    
    # Crucial: Close the internal figure generated by sklearn to save memory
    plt.close(display.figure_)
    
    return pdp_values, grid_values
 
def _pdp_worker(seed, feature_name, feature_idx, target_grid, model_name):
    try:
        model, X = load_data_raw(seed, model_name)
        if model is None: return None
        
        # --- THE WORKING LOGIC ---
        # We use from_estimator because we know it works with your sklearn version.
        display = PartialDependenceDisplay.from_estimator(
            model, 
            X, 
            features=[feature_idx], # Pass the INTEGER index
            kind='average', 
            grid_resolution=50,
            ax=None 
        )
        
        # Extract Y (Prediction) - Scikit-learn structure
        # 'average' is usually a list of arrays (one for each feature requested)
        # We requested 1 feature, so we take index [0]
        y_native = display.pd_results[0]['average'][0]
        
        # Extract X (Grid) - Robust extraction from the plot lines
        # This matches your "previews code" logic which is safest
        x_native = display.axes_[0, 0].lines[0].get_xdata()
        
        # Cleanup memory immediately
        plt.close(display.figure_)
        
        # Interpolate to match the Global Grid (so we can average later)
        y_interpolated = np.interp(target_grid, x_native, y_native)
        
        return pd.DataFrame({
            'feature': feature_name, # Save the readable string name
            'seed': seed,
            'grid_value': target_grid,
            'pdp_value': y_interpolated
        })
        
    except Exception as e:
        # If it fails, we print precisely why
        return f"Error (S{seed} {feature_name}): {e}"
    
def compute_pdp_safe_production(
    model_name, 
    seeds, 
    tech_features_list, 
    save_dir, 
    grid_resolution=50
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Define Global Grids (Scan Data)
    print(f"--- 1. Defining Global Grids for {len(tech_features_list)} features ---")
    global_grids = {}
    feat_to_idx = {name: i for i, name in enumerate(tech_features_list)}
    
    for feature in tqdm(tech_features_list, desc="Scanning"):
        idx = feat_to_idx[feature]
        all_vals = []
        # Check first 2 seeds
        for seed in list(seeds)[:2]:
            _, X = load_data_raw(seed, model_name)
            if X is not None:
                # Get column by integer index
                all_vals.append(X.iloc[:, idx].values)
        
        if all_vals:
            flat = np.concatenate(all_vals)
            mi, ma = np.percentile(flat, [1, 99])
            if mi == ma: ma += 1e-9
            global_grids[feature] = np.linspace(mi, ma, grid_resolution)
            
    # 2. Parallel Computation
    print(f"--- 2. Computing PDPs ---")
    tasks = []
    for feature in global_grids:
        grid = global_grids[feature]
        idx = feat_to_idx[feature]
        for seed in seeds:
            tasks.append((seed, feature, idx, grid, model_name))
            
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_pdp_worker)(s, f, i, g, m) for s, f, i, g, m in tasks
    )
    
    # 3. Process Results
    valid = [r for r in results if isinstance(r, pd.DataFrame)]
    errors = [r for r in results if isinstance(r, str)]
    
    if errors:
        print(f"⚠️ Warning: {len(errors)} failures. First error: {errors[0]}")
        
    if not valid:
        print("❌ Fatal: No results generated.")
        return None
        
    full_df = pd.concat(valid, ignore_index=True)
    summary_df = full_df.groupby(['feature', 'grid_value'])['pdp_value'].agg(['mean', 'std']).reset_index()
    
    # Save
    full_df.to_csv(save_dir / f"{model_name}_PDP_Raw_Long.csv", index=False)
    summary_df.to_csv(save_dir / f"{model_name}_PDP_Summary.csv", index=False)
    
    print(f"✅ SUCCESS. Saved to {save_dir}")
    return summary_df

# ________________________________________________________________________
# Main Figures Functions

def plot_combined_scientific_figure(
    importance_df,  
    model_name,    
    models,            
    X_test_dfs,        
    pdp_features,      
    n_of_perm_imp_to_plot: int = 10,
    save_path=None
):
    """
    Creates a single publication-ready figure containing:
    (A) Permutation Feature Importance
    (B) Robust Partial Dependence Plots (Grid)
    """
    
    if not isinstance(importance_df, pd.DataFrame):
        raise ValueError(f"Expected importance_df to be a DataFrame, got {type(importance_df)}.")

    # --- Setup Figure Layout ---
    fig = plt.figure(figsize=(14, 18), dpi=300) 
    
    # 1. OUTER LAYOUT
    gs_top = fig.add_gridspec(nrows=1, ncols=1, 
                              left=0.15, right=0.9, top=0.9, bottom=0.65)

    n_pdp = len(pdp_features)
    n_cols = 3
    n_rows = math.ceil(n_pdp / n_cols)

    gs_bot = fig.add_gridspec(nrows=n_rows, ncols=n_cols, 
                              left=0.1, right=0.9, top=0.5, bottom=0.1,
                              wspace=0.3, hspace=0.3)
    
    # ===================================================
    # PANEL A: Permutation Feature Importance
    # ===================================================
    with sns.axes_style("whitegrid"):
        ax_imp = fig.add_subplot(gs_top[0])
        plot_df = importance_df.sort_values('mean_importance', ascending=False).tail(n_of_perm_imp_to_plot)
        
        sns.barplot(data=plot_df, x='mean_importance', y=plot_df.index, color='skyblue', ax=ax_imp)
        
        ax_imp.errorbar(
            x=plot_df['mean_importance'], y=np.arange(len(plot_df)), 
            xerr=plot_df['std_importance'], fmt='none', c='black', capsize=5
        )

        ax_imp.set_title(f'{model_name} Permutation Feature Importance (Top {n_of_perm_imp_to_plot} Aggregated)', fontsize=18, weight='bold')
        ax_imp.set_xlabel('Mean Importance Score', fontsize=12, fontweight ='bold')
        ax_imp.set_ylabel('', fontsize=12)

    # ===================================================
    # PANEL B: Robust PDP Grid
    # ===================================================
    print(f"--- Calculating PDPs for {len(pdp_features)} features across {len(models)} seeds ---")

    for i, feature in enumerate(pdp_features):
        row = i // n_cols
        col = i % n_cols
        ax_pdp = fig.add_subplot(gs_bot[row, col])
        
        tasks = [delayed(_get_pdp_data)(model, X_test, feature) 
                 for model, X_test in zip(models, X_test_dfs)]
        results = Parallel(n_jobs=-1)(tasks)
        
        all_vals = np.array([res[0] for res in results])
        grid_vals = results[0][1]
        
        mean_pdp = np.mean(all_vals, axis=0)
        std_pdp = np.std(all_vals, axis=0)
        
        line, = ax_pdp.plot(grid_vals, mean_pdp, color='#C44E52', linewidth=2.5)
        fill = ax_pdp.fill_between(grid_vals, mean_pdp - std_pdp, mean_pdp + std_pdp, 
                                   color='#C44E52', alpha=0.15)
        
        ax_pdp.set_xlabel(feature, fontsize=11, fontweight='bold')
        if col == 0:
            ax_pdp.set_ylabel("Partial Dependence", fontsize=11, fontweight ='bold')
        ax_pdp.grid(True, linestyle=':', alpha=0.6)

    # ===================================================
    # LABELS & LEGEND
    # ===================================================
    fig.text(0.02, 0.96, 'A', fontsize=26, fontweight='bold', va='top')
    fig.text(0.02, 0.6, 'B', fontsize=26, fontweight='bold', va='top')
    fig.text(0.5, 0.55, f'{model_name} Partial Dependence Plots', 
             fontsize=18, fontweight='bold', ha='center', va='top')

    fig.legend([line, fill], ["Mean PDP across seeds", "±1 Std. Dev."], 
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.05), 
               fontsize=12, frameon=False)
    
    # ===================================================
    # SAVE LOGIC (Inserted Here)
    # ===================================================
    if save_path:
        # Ensure the folder exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # Save the figure
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Combined Figure saved to: {save_path}")

    plt.show()

def plot_combined_figure_decoupled(
    perm_summary_path,
    perm_raw_path,
    pdp_summary_path,
    mapping_dict,
    save_path=None,
    top_n=10,
    custom_pdp_features=None,
    
    # --- Styling ---
    font_size=16,
    font_size_b=15,
    label_pad=10,
    
    # --- Layout Controls ---
    margin_left_A=0.28,   
    margin_right_A=0.95,

    margin_left_B=0.10,   
    margin_right_B=0.95,
    
    split_y=0.70, 
    
    panel_b_wspace=0.3,
    panel_b_hspace=0.5,
    show_=False,
    save = False

):
    # --- 0. Setup ---
    set_nature_style(font_size=font_size)
    PUB_TO_TECH = {v: k for k, v in mapping_dict.items()}
    
    perm_path = Path(perm_summary_path)
    raw_path = Path(perm_raw_path)
    pdp_path = Path(pdp_summary_path)
    
    if not perm_path.exists() or not pdp_path.exists():
        print(f"❌ Error: Data files missing.")
        return

    perm_df = pd.read_csv(perm_path, index_col=0)
    if raw_path.exists(): raw_df = pd.read_csv(raw_path, index_col=0)
    else: raw_df = None
    pdp_df = pd.read_csv(pdp_path)

    # ==========================================================================
    # PREPARE DATA
    # ==========================================================================
    df = perm_df.copy()
    
    if 'mean_importance' not in df.columns:
        if 'mean_importance' in (df.index.name or ''): df = df.reset_index()
        else:
            cols = df.columns.tolist()
            rename_dict = {}
            if len(cols) >= 2: rename_dict[cols[0]] = 'Feature'; rename_dict[cols[1]] = 'mean_importance'
            if len(cols) >= 3: rename_dict[cols[2]] = 'std_importance'
            df = df.rename(columns=rename_dict)
            if 'Feature' in df.columns: df.set_index('Feature', inplace=True)

    df = df.sort_values(by='mean_importance', ascending=True).tail(top_n)
    
    # Map Names
    first_idx = df.index[0]
    is_pub_name = first_idx in mapping_dict.values()
    if is_pub_name:
        df['DisplayName'] = df.index
        df['TechName'] = [PUB_TO_TECH.get(x, x) for x in df.index]
    else:
        df['DisplayName'] = [mapping_dict.get(x, x) for x in df.index]
        df['TechName'] = df.index
    
    top_features_ordered = df.index.tolist()[::-1]

    # ==========================================================================
    # INITIALIZE FIGURE
    # ==========================================================================
    fig = plt.figure(figsize=(14, 18), dpi=300)
    
    # Grid A: Top Panel (Ends at split_y + gap)
    # We add +0.05 to bottom of A to create the gap you requested
    gs_top = fig.add_gridspec(
        nrows=1, ncols=1,
        left=margin_left_A, right=margin_right_A,
        top=0.95, bottom=split_y + 0.08 
    )

    # Grid B: Bottom Panel (Starts at split_y - gap)
    gs_bot = fig.add_gridspec(
        nrows=2, ncols=3,
        left=margin_left_B, right=margin_right_B, 
        top=split_y - 0.05, bottom=0.08,
        wspace=panel_b_wspace, hspace=panel_b_hspace
    )
    
    # ==========================================================================
    # PANEL A: FEATURE IMPORTANCE
    # ==========================================================================
    ax_imp = fig.add_subplot(gs_top[0])
    
    sns.barplot(
        x='mean_importance', y='DisplayName', data=df, 
        order=top_features_ordered, color='skyblue', edgecolor='black', linewidth=1.2,
        ax=ax_imp, zorder=2
    )

    if 'std_importance' in df.columns:
        df_sorted = df.reindex(top_features_ordered)
        ax_imp.errorbar(
            x=df_sorted['mean_importance'], y=np.arange(len(df_sorted)), 
            xerr=df_sorted['std_importance'], fmt='none', c='black', capsize=6, linewidth=1.5, zorder=4
        )
        
    # --- POINTS OVERLAY (FIXED) ---
    if raw_df is not None:
        # Check if Raw DF uses Publication Names or Technical Names
        raw_cols = set(raw_df.columns)
        target_tech = set(df['TechName'])
        target_disp = set(df['DisplayName'])
        
        melted = None
        
        # 1. Try matching against Publication Names (Most likely scenario if using previous code)
        valid_cols_disp = raw_cols.intersection(target_disp)
        if len(valid_cols_disp) > 0:
            melted = raw_df[list(valid_cols_disp)].melt(var_name='DisplayName', value_name='Importance')
            
        # 2. If not found, try matching Technical Names
        elif len(raw_cols.intersection(target_tech)) > 0:
            valid_cols_tech = list(raw_cols.intersection(target_tech))
            melted = raw_df[valid_cols_tech].melt(var_name='TechName', value_name='Importance')
            # Map Tech -> Display
            tech_map = dict(zip(df['TechName'], df['DisplayName']))
            melted['DisplayName'] = melted['TechName'].map(tech_map)
            
        if melted is not None:
            sns.stripplot(
                x='Importance', y='DisplayName', data=melted,
                order=top_features_ordered,
                color='#C44E52',  size=5, jitter=0.25,
                ax=ax_imp, zorder=10, linewidth=0.5, edgecolor='black'
            )

    # Aesthetics A
    ax_imp.spines['left'].set_visible(True)
    ax_imp.spines['bottom'].set_visible(True)
    ax_imp.spines['left'].set_linewidth(1.5)
    ax_imp.spines['bottom'].set_linewidth(1.5)
    
    ax_imp.tick_params(axis='both', which='major', bottom=True, left=True, direction='out', length=6, width=1.5, labelsize=font_size)
    ax_imp.xaxis.set_minor_locator(AutoMinorLocator())
    ax_imp.tick_params(axis='x', which='minor', bottom=True, direction='out', length=3, width=1.0)
    ax_imp.grid(False)
    ax_imp.set_xlabel('Relative Mean Importance Score', fontsize=font_size + 2, labelpad=label_pad-10)
    ax_imp.set_ylabel('')

    # ==========================================================================
    # PANEL B: PDP GRID (Decoupled)
    # ==========================================================================
    if custom_pdp_features: pdp_tech_names = custom_pdp_features
    else: pdp_tech_names = df['TechName'].tolist()[::-1][:6]

    handles = []
    
    for i, tech_feat in enumerate(pdp_tech_names):
        # Calculate grid position manually for the 2x3 grid
        row, col = i // 3, i % 3
        
        # Safety check if user asks for >6 features
        if row >= 2: break 
            
        ax = fig.add_subplot(gs_bot[row, col])
        
        subset = pdp_df[pdp_df['feature'] == tech_feat]
        
        if not subset.empty:
            subset = subset.sort_values('grid_value')
            l, = ax.plot(subset['grid_value'], subset['mean'], color='black', linewidth=2.5)
            f = ax.fill_between(
                subset['grid_value'],
                subset['mean'] - subset['std'],
                subset['mean'] + subset['std'],
                color='skyblue', alpha=0.5
            )
            if i == 0: handles = [l, f]
            
        pub_name = mapping_dict.get(tech_feat, tech_feat)
        ax.set_xlabel(pub_name, fontweight='bold', fontsize=font_size_b)
        
        ax.xaxis.set_major_locator(MultipleLocator(1))
        
        if col == 0: 
            ax.set_ylabel("Partial Dependence", fontsize=font_size, labelpad=label_pad, fontweight='bold')
            
        ax.tick_params(axis='both', which='major', bottom=True, left=True, direction='out', length=5, width=1.2, labelsize=font_size-2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.grid(True, linestyle=':', alpha=0.6)

    # ==========================================================================
    # FINAL LAYOUT & LABELS
    # ==========================================================================
    # Adjusted coordinates based on your user input
    fig.text(0.02, 0.98, 'A.                            Features Permutation Importance', fontsize=24, fontweight='bold')
    
    # B Label roughly centered in the gap
    fig.text(0.02, split_y, 'B.                                     Partial Dependence Plots', fontsize=24, fontweight='bold', va='center')
    
    if handles:
        fig.legend(handles, ['Mean Effect', '±1 Std. Dev.'], loc='lower center', 
                   bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=font_size)
    
    if save:
        if save_path:
            plt.savefig(save_path,dpi=IMAGE_DPI)
            print(f"✅ Saved figure to: {save_path}")
    
    if show_:
        plt.show()
    else:
        plt.close()

def plot_combined_figure_side_by_side(
    perm_summary_path,
    perm_raw_path,
    pdp_summary_path,
    mapping_dict,
    save_path=None,
    top_n=10,
    custom_pdp_features=None,
    
    # --- Styling ---
    font_size=16,
    font_size_b=15,
    label_pad=10,
    
    # --- Layout Controls (Updated for Side-by-Side) ---
    margin_left_A=0.08,   
    margin_right_A=0.40,  

    margin_left_B=0.50,   
    margin_right_B=0.95,
    
    top_margin=0.88,
    bottom_margin=0.15,
    
    panel_b_wspace=0.35,
    panel_b_hspace=0.45,
    show_=False,
    save=False
):
    # --- 0. Setup ---
    # set_nature_style(font_size=font_size) # Uncomment if using your custom style function
    PUB_TO_TECH = {v: k for k, v in mapping_dict.items()}
    
    perm_path = Path(perm_summary_path)
    raw_path = Path(perm_raw_path)
    pdp_path = Path(pdp_summary_path)
    
    if not perm_path.exists() or not pdp_path.exists():
        print(f"❌ Error: Data files missing.")
        return

    perm_df = pd.read_csv(perm_path, index_col=0)
    if raw_path.exists(): raw_df = pd.read_csv(raw_path, index_col=0)
    else: raw_df = None
    pdp_df = pd.read_csv(pdp_path)

    # ==========================================================================
    # PREPARE DATA
    # ==========================================================================
    df = perm_df.copy()
    
    if 'mean_importance' not in df.columns:
        if 'mean_importance' in (df.index.name or ''): df = df.reset_index()
        else:
            cols = df.columns.tolist()
            rename_dict = {}
            if len(cols) >= 2: rename_dict[cols[0]] = 'Feature'; rename_dict[cols[1]] = 'mean_importance'
            if len(cols) >= 3: rename_dict[cols[2]] = 'std_importance'
            df = df.rename(columns=rename_dict)
            if 'Feature' in df.columns: df.set_index('Feature', inplace=True)

    df = df.sort_values(by='mean_importance', ascending=True).tail(top_n)
    
    # Map Names
    first_idx = df.index[0]
    is_pub_name = first_idx in mapping_dict.values()
    if is_pub_name:
        df['DisplayName'] = df.index
        df['TechName'] = [PUB_TO_TECH.get(x, x) for x in df.index]
    else:
        df['DisplayName'] = [mapping_dict.get(x, x) for x in df.index]
        df['TechName'] = df.index
    
    top_features_ordered = df.index.tolist()[::-1]

    # ==========================================================================
    # INITIALIZE FIGURE (Wider configuration for side-by-side)
    # ==========================================================================
    fig = plt.figure(figsize=(20, 12), dpi=300)
    
    # Grid A: Left Panel (Vertical Bar Chart)
    gs_left = fig.add_gridspec(
        nrows=1, ncols=1,
        left=margin_left_A, right=margin_right_A,
        top=top_margin, bottom=bottom_margin 
    )

    # Grid B: Right Panel (3 Rows x 2 Columns Matrix)
    gs_right = fig.add_gridspec(
        nrows=3, ncols=2,
        left=margin_left_B, right=margin_right_B, 
        top=top_margin, bottom=bottom_margin,
        wspace=panel_b_wspace, hspace=panel_b_hspace
    )
    
    # ==========================================================================
    # PANEL A: FEATURE IMPORTANCE (Vertical Bars)
    # ==========================================================================
    ax_imp = fig.add_subplot(gs_left[0])
    
    sns.barplot(
        x='DisplayName', y='mean_importance', data=df, 
        order=top_features_ordered, color='skyblue', edgecolor='black', linewidth=1.2,
        ax=ax_imp, zorder=2
    )

    if 'std_importance' in df.columns:
        df_sorted = df.reindex(top_features_ordered)
        ax_imp.errorbar(
            x=np.arange(len(df_sorted)), y=df_sorted['mean_importance'], 
            yerr=df_sorted['std_importance'], fmt='none', c='black', capsize=6, linewidth=1.5, zorder=4
        )
        
    # --- POINTS OVERLAY ---
    if raw_df is not None:
        raw_cols = set(raw_df.columns)
        target_tech = set(df['TechName'])
        target_disp = set(df['DisplayName'])
        
        melted = None
        
        valid_cols_disp = raw_cols.intersection(target_disp)
        if len(valid_cols_disp) > 0:
            melted = raw_df[list(valid_cols_disp)].melt(var_name='DisplayName', value_name='Importance')
            
        elif len(raw_cols.intersection(target_tech)) > 0:
            valid_cols_tech = list(raw_cols.intersection(target_tech))
            melted = raw_df[valid_cols_tech].melt(var_name='TechName', value_name='Importance')
            tech_map = dict(zip(df['TechName'], df['DisplayName']))
            melted['DisplayName'] = melted['TechName'].map(tech_map)
            
        if melted is not None:
            sns.stripplot(
                x='DisplayName', y='Importance', data=melted,
                order=top_features_ordered,
                color='#C44E52',  size=5, jitter=0.25,
                ax=ax_imp, zorder=10, linewidth=0.5, edgecolor='black'
            )

    # Aesthetics A
    ax_imp.spines['right'].set_visible(False)
    ax_imp.spines['top'].set_visible(False)
    ax_imp.spines['left'].set_linewidth(1.5)
    ax_imp.spines['bottom'].set_linewidth(1.5)
    
    ax_imp.tick_params(axis='both', which='major', bottom=True, left=True, direction='out', length=6, width=1.5, labelsize=font_size)
    
    # Rotate X-axis labels to 45 degrees so they fit nicely
    ax_imp.set_xticklabels(ax_imp.get_xticklabels(), rotation=45, ha='right')
    
    ax_imp.yaxis.set_minor_locator(AutoMinorLocator())
    ax_imp.tick_params(axis='y', which='minor', left=True, direction='out', length=3, width=1.0)
    ax_imp.grid(False)
    
    ax_imp.set_ylabel('Relative Mean Importance Score', fontsize=font_size + 2, labelpad=label_pad)
    ax_imp.set_xlabel('')

    # ==========================================================================
    # PANEL B: PDP GRID (3x2 Matrix, Standard X/Y Orientation)
    # ==========================================================================
    if custom_pdp_features: pdp_tech_names = custom_pdp_features
    else: pdp_tech_names = df['TechName'].tolist()[::-1][:6]

    handles = []
    
    for i, tech_feat in enumerate(pdp_tech_names):
        # 3 Rows x 2 Columns mapping
        row, col = i // 2, i % 2
        
        # Stop if we exceed 6 plots (3x2 matrix limit)
        if row >= 3: break 
            
        ax = fig.add_subplot(gs_right[row, col])
        
        subset = pdp_df[pdp_df['feature'] == tech_feat]
        
        if not subset.empty:
            subset = subset.sort_values('grid_value')
            # Standard Orientation (X = grid_value, Y = mean)
            l, = ax.plot(subset['grid_value'], subset['mean'], color='black', linewidth=2.5)
            f = ax.fill_between(
                subset['grid_value'],
                subset['mean'] - subset['std'],
                subset['mean'] + subset['std'],
                color='skyblue', alpha=0.5
            )
            if i == 0: handles = [l, f]
            
        pub_name = mapping_dict.get(tech_feat, tech_feat)
        
        # Feature name is back on the X-axis
        ax.set_xlabel(pub_name, fontweight='bold', fontsize=font_size_b)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        
        # Partial Dependence label on the left column (col == 0)
        if col == 0: 
            ax.set_ylabel("Partial Dependence", fontsize=font_size, labelpad=label_pad, fontweight='bold')
            
        ax.tick_params(axis='both', which='major', bottom=True, left=True, direction='out', length=5, width=1.2, labelsize=font_size-2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True, linestyle=':', alpha=0.6)

    # ==========================================================================
    # FINAL LAYOUT & LABELS
    # ==========================================================================
    # Position A over the left pane, and B over the right pane
    fig.text(margin_left_A, 0.95, 'A.  Features Permutation Importance', fontsize=24, fontweight='bold')
    fig.text(margin_left_B, 0.95, 'B.  Partial Dependence Plots', fontsize=24, fontweight='bold')
    
    # Legend centered under Panel B
    if handles:
        fig.legend(handles, ['Mean Effect', '±1 Std. Dev.'], loc='lower center', 
                   bbox_to_anchor=(margin_left_B + (margin_right_B - margin_left_B)/2, 0.02), 
                   ncol=2, frameon=False, fontsize=font_size)
    
    if save:
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved figure to: {save_path}")
    
    if show_:
        plt.show()
    else:
        plt.close()



# ==============================================================================
# 2. Prediction Surface Manifold of Models
# ==============================================================================

def plot_robust_3d_landscape(
        models, 
        global_X_scaled, 
        global_y, 
        resolution=50, 
        scatter=False, 
        xylabelsize=20, 
        xyz_tick_labelsize=20, 
        cbar_tick_size=18, 
        cbar_label_size=20, 
        save_path=None
    ):
    """
    Visualizes the ensemble decision surface using a custom Alpha-to-Beta color gradient.
    Caches the computed mesh and scatter data to avoiding re-running PCA/Predictions.
    """
    set_nature_style()

    alpha_color = COLOR_BLIND_PALETTE['orange'] 
    beta_color  = COLOR_BLIND_PALETTE['blue'] 
    mid_color   = 'lightgray'

    custom_cmap = LinearSegmentedColormap.from_list(
        "AlphaGrayBeta", 
        [alpha_color, mid_color, beta_color]
    )

    
    # --- 1. Caching Setup ---
    data_loaded = False
    cache_path = None
    
    if save_path:
        # Create a sibling file with .npz extension for data
        save_path_obj = Path(save_path)
        cache_path = save_path_obj.parent / f"{save_path_obj.stem}_data.npz"
        
        if cache_path.exists():
            print(f"--- 🟢 Loading Cached 3D Landscape Data from: {cache_path.name} ---")
            try:
                data = np.load(cache_path)
                xx = data['xx']
                yy = data['yy']
                Z = data['Z']
                X_pca_global = data['X_pca_global']
                avg_actual_probs = data['avg_actual_probs']
                pca_ratios = data['pca_ratios']
                data_loaded = True
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}. Recomputing...")

    if not data_loaded:
        print("--- 🟡 Computing Robust 3D Ensemble Landscape ---")
        
        # A. Fit PCA
        print("   Fitting PCA...")
        pca = PCA(n_components=2)
        X_pca_global = pca.fit_transform(global_X_scaled)
        pca_ratios = pca.explained_variance_ratio_
        
        # B. Meshgrid
        x_min, x_max = X_pca_global[:, 0].min() - 1, X_pca_global[:, 0].max() + 1
        y_min, y_max = X_pca_global[:, 1].min() - 1, X_pca_global[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        
        # C. Inverse Transform -> Predict
        print(f"   Querying ensemble surface ({len(models)} models)...")
        grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_synthetic_scaled = pca.inverse_transform(grid_points_2d)
        
        feature_names = global_X_scaled.columns.tolist()
        grid_df = pd.DataFrame(grid_synthetic_scaled, columns=feature_names)
        
        all_probs = []
        for model in models:
            probs = model.predict_proba(grid_df)[:, 1]
            all_probs.append(probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        Z = avg_probs.reshape(xx.shape)
        
        # D. Get actual probabilities for scatter height
        all_actual_probs = []
        for model in models:
            all_actual_probs.append(model.predict_proba(global_X_scaled)[:, 1])
        avg_actual_probs = np.mean(all_actual_probs, axis=0)
        
        # E. Save Data
        if cache_path:
            np.savez_compressed(
                cache_path,
                xx=xx, yy=yy, Z=Z,
                X_pca_global=X_pca_global,
                avg_actual_probs=avg_actual_probs,
                pca_ratios=pca_ratios
            )
            print(f"✅ Data saved to: {cache_path}")

    # ==========================================
    # PLOTTING
    # ==========================================
    fig = plt.figure(figsize=(12, 10), dpi=300) # Nature standard high res
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface
    surf = ax.plot_surface(xx, yy, Z, cmap=custom_cmap, alpha=0.8, 
                           edgecolor='none', antialiased=True, vmin=0, vmax=1)
    
    if scatter:
        mask_alpha = (global_y == 0)
        mask_beta  = (global_y == 1)
        
        # Alpha Points (Class 0)
        ax.scatter(X_pca_global[mask_alpha, 0], X_pca_global[mask_alpha, 1], avg_actual_probs[mask_alpha],
                   c=alpha_color, marker='x', s=20, alpha=0.3, label='True Alpha', depthshade=False)
        
        # Beta Points (Class 1)
        ax.scatter(X_pca_global[mask_beta, 0], X_pca_global[mask_beta, 1], avg_actual_probs[mask_beta],
                   c=beta_color, marker='o', s=20, alpha=0.3, label='True Beta', depthshade=False)

    # Formatting (Nature Style Fonts)
    ax.set_xlabel(f"PC1 ({pca_ratios[0]:.1%} var)", fontsize=xylabelsize, labelpad=10, fontname='serif')
    ax.set_ylabel(f"PC2 ({pca_ratios[1]:.1%} var)", fontsize=xylabelsize, labelpad=10, fontname='serif')
    ax.set_zlabel("Probability (Beta)", fontsize=xylabelsize, labelpad=10, fontname='serif')
    ax.set_zlim(0, 1.0)

    # Ticks
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_tick_params(labelsize=xyz_tick_labelsize)
        for label in axis.get_ticklabels():
            label.set_fontname('serif')

    # View Angle
    ax.view_init(elev=35, azim=230)
    
    # Colorbar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label('Probability: Alpha → Beta', fontsize=cbar_label_size, labelpad=15, fontname='serif')
    cbar.ax.tick_params(labelsize=cbar_tick_size)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontname('serif')

    # Remove pane fills for cleaner look (Nature style usually cleaner)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=IMAGE_DPI, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
        
    plt.show()

def plot_gradient_histogram(
    models, 
    X_scaled_list, 
    save_path=None,
    # --- Styling ---
    n_bins=30,
    font_size=14
):
    """
    Plots a histogram of predicted probabilities with colored bars.
    Includes explicit ticks on X and Y axes.
    """
    set_nature_style(font_size=12)

    alpha_color = COLOR_BLIND_PALETTE['orange'] 
    beta_color  = COLOR_BLIND_PALETTE['blue'] 
    mid_color   = 'lightgray'

    custom_cmap = LinearSegmentedColormap.from_list(
        "AlphaGrayBeta", 
        [alpha_color, mid_color, beta_color]
    )

    # --- 1. Caching Logic ---
    all_probs = None
    cache_path = None
    
    if save_path:
        save_path_obj = Path(save_path)
        cache_path = save_path_obj.parent / f"{save_path_obj.stem}_data.csv"
        
        if cache_path.exists():
            print(f"--- 🟢 Loading Cached Probabilities from: {cache_path.name} ---")
            try:
                df_cache = pd.read_csv(cache_path)
                all_probs = df_cache['probability'].values
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}. Recomputing...")

    if all_probs is None:
        print("--- 🟡 Calculating Predictions for Histogram ---")
        temp_probs = []
        for model, X in zip(models, X_scaled_list):
            preds = model.predict_proba(X)[:, 1].ravel()
            temp_probs.extend(preds)
        
        all_probs = np.array(temp_probs)
        
        if cache_path:
            pd.DataFrame({'probability': all_probs}).to_csv(cache_path, index=False)
            print(f"✅ Probabilities saved to: {cache_path}")

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 6), dpi=IMAGE_DPI)
    ax = plt.gca() # Get current axis to apply ticks
    
    # Histogram
    counts, bins, patches = plt.hist(
        all_probs, bins=n_bins, 
        edgecolor='white', linewidth=0.5, alpha=0.9
    )
    
    # Color Bars
    for bin_left, patch in zip(bins, patches):
        bin_center = bin_left + (bins[1] - bins[0]) / 2
        plt.setp(patch, 'facecolor', custom_cmap(bin_center))

    # Labels
    plt.xlabel("Predicted Probability (Beta)", fontsize=font_size+2, labelpad=10)
    plt.ylabel("Count of Cells", fontsize=font_size+2, labelpad=10)
    plt.xlim(0, 1)
    
    # --- TICKS CONFIGURATION (The Request) ---
    # Force ticks visible on bottom and left
    ax.tick_params(
        axis='both',       # Apply to X and Y
        which='major',     # Major ticks
        direction='out',   # Point outward
        bottom=True,       # Force X ticks
        left=True,         # Force Y ticks
        length=6,          # Visible length
        width=1.2,         # Visible width
        labelsize=font_size
    )
    
    # Add minor ticks to X axis for precision
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='x', which='minor', bottom=True, direction='out', length=3, width=0.8)

    # Grid (Light)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    # Annotations
    max_h = max(counts)
    
    plt.text(0.05, max_h, "Alpha Basin\n(Low Probability)", 
             color=alpha_color, weight='bold', ha='left', va='top', fontsize=font_size)
    
    plt.text(0.95, max_h, "Beta Plateau\n(High Probability)", 
             color=beta_color, weight='bold', ha='right', va='top', fontsize=font_size)
    
    plt.text(0.5, max_h, "Zone of\nUncertainty", 
             color='gray', ha='center', va='top', fontsize=font_size, alpha=0.8)
    
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.6, ymax=0.85)

    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=IMAGE_DPI)
        print(f"✅ Figure saved to: {save_path}")
        
    plt.show()

def combine_images_on_canvas_aligned(
    image_path_1, 
    image_path_2, 
    save_path=None, 
    title="Combined Analysis",
    # --- Styling ---
    spacer_width=0.05, 
    fontsize_title=26,
    fontsize_subtitle=24
):
    """
    Loads two existing image files and displays them side-by-side with 
    perfectly aligned subtitles using Nature-style fonts.
    """
    # Apply Style
    set_nature_style()
    
    # 1. Check Files
    img1_path = Path(image_path_1)
    img2_path = Path(image_path_2)
    
    if not img1_path.exists():
        print(f"❌ Error: Image 1 not found: {img1_path}")
        return
    if not img2_path.exists():
        print(f"❌ Error: Image 2 not found: {img2_path}")
        return

    # 2. Load Images
    try:
        img1 = mpimg.imread(str(img1_path))
        img2 = mpimg.imread(str(img2_path))
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return

    # 3. Create Figure
    # constrained_layout=True helps, but for manual text placement we sometimes disable it 
    # to have absolute control. Here we keep it but rely on figure coordinates for text.
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), 
                             gridspec_kw={'wspace': spacer_width}, 
                             constrained_layout=False) # False gives us more control over exact placement
    
    # 4. Display Images
    axes[0].imshow(img1)
    axes[0].axis('off') 

    axes[1].imshow(img2)
    axes[1].axis('off')

    # --- 5. Manual Alignment of Subtitles ---
    
    # Trigger draw to calculate positions
    fig.canvas.draw()

    # Get positions (0 to 1 coordinates)
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()

    # Calculate Centers
    x0_center = (pos0.x0 + pos0.x1) / 2
    x1_center = (pos1.x0 + pos1.x1) / 2

    # Vertical Position (Just above images)
    # We use the max y-coordinate of the axes plus a small buffer
    subtitle_y = max(pos0.y1, pos1.y1) + 0.02

    # Place Labels
    fig.text(x0_center, subtitle_y, "A. 3D Averaged Decision Surface",
             fontsize=fontsize_subtitle, weight='bold', ha='center', va='bottom', fontname='serif')

    fig.text(x1_center, subtitle_y, "B. Distribution of Ensemble Predictions",
             fontsize=fontsize_subtitle, weight='bold', ha='center', va='bottom', fontname='serif')

    # Overall Title
    if title:
        fig.suptitle(title, fontsize=fontsize_title, weight='bold', y=subtitle_y + 0.08, fontname='serif')

    # 6. Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path,dpi=IMAGE_DPI, bbox_inches='tight') 
        print(f"✅ Combined image saved to: {save_path}")

    plt.show()




# ==============================================================================
# 2. SHAP Analysis
# ==============================================================================

alpha_color = COLOR_BLIND_PALETTE['orange'] 
beta_color  = COLOR_BLIND_PALETTE['blue'] 
mid_color   = 'lightgray'

custom_shap_cmap = LinearSegmentedColormap.from_list(
    "AlphaGrayBeta", 
    [alpha_color, mid_color, beta_color]
)

def compute_and_save_shap_data(
    model_name, 
    seeds, 
    tech_names,    # <--- 1. ADDED THIS PARAMETER
    save_dir, 
    n_samples_per_seed=255
):
    """
    Computes SHAP values across all seeds, aggregates them, and saves to disk.
    Returns: (shap_values, X_data)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define filenames
    shap_file = save_dir / f"{model_name}_SHAP_Values_Global.npy"
    x_file    = save_dir / f"{model_name}_SHAP_X_Global.csv"
    
    # --- 1. Check Cache ---
    if shap_file.exists() and x_file.exists():
        print(f"--- 🟢 Loading Cached SHAP Data ---")
        print(f"    SHAP: {shap_file.name}")
        print(f"    Data: {x_file.name}")
        
        # Load efficiently
        global_shap_values = np.load(shap_file)
        global_X = pd.read_csv(x_file)
        return global_shap_values, global_X

    # --- 2. Compute (If Cache Missing) ---
    print(f"--- 🟡 Computing SHAP for {model_name} (this may take time) ---")
    
    shap_list = []
    X_list = []
    
    for seed in tqdm(seeds, desc="SHAP Computation"):
        # --- 2. PASSED tech_names HERE TO RESTORE COLUMNS ---
        X_test, _ = load_data_for_seed(RESULTS_ROOT, seed, 'test', feature_names=tech_names)
        model_path = RESULTS_ROOT / f"Results_{seed}/Results/Final_Results/Test_Results/{model_name}_final_model.pkl"
        
        if X_test is None or not model_path.exists(): continue
        
        try:
            model = joblib.load(model_path)
            
            # Subsample if needed
            if len(X_test) > n_samples_per_seed:
                X_subset = X_test.sample(n=n_samples_per_seed, random_state=seed)
            else:
                X_subset = X_test
                
            # Create Explainer (TreeExplainer is fastest for LGBM)
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_subset)
            
            # Handle Binary Classification Output (SHAP returns list [class0, class1])
            if isinstance(shap_vals, list): 
                target_shap = shap_vals[1] # Class 1 (Beta)
            elif len(shap_vals.shape) == 3:
                target_shap = shap_vals[:, :, 1]
            else:
                target_shap = shap_vals
                
            shap_list.append(target_shap)
            X_list.append(X_subset)
            
        except Exception as e:
            print(f"Error seed {seed}: {e}")

    if not shap_list:
        print("❌ Error: No SHAP values computed.")
        return None, None

    # Aggregate
    global_shap_values = np.vstack(shap_list)
    global_X = pd.concat(X_list, axis=0, ignore_index=True)
    
    # --- 3. Save ---
    np.save(shap_file, global_shap_values)
    global_X.to_csv(x_file, index=False)
    
    print(f"✅ Saved Aggregated SHAP Data ({len(global_X)} samples)")
    return global_shap_values, global_X

def plot_shap_nature_style(
    model_name,
    shap_values,
    X_df,
    mapping_dict,
    save_path=None,
    # --- Styling ---
    font_family='serif',
    title_size=26,
    label_size=22,
    tick_size=18
):
    """
    Generates a publication-ready SHAP summary figure (Beeswarm + 2 Dep Plots).
    Uses cached data and applies strict styling.
    """
    # 1. Apply Nature Style
    set_nature_style()
    
    # 2. Rename Data for Display (Technical -> Publication)
    X_display = X_df.rename(columns=mapping_dict)
    
    # 3. Identify Top Features Automatically
    # Mean absolute SHAP value across all samples
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1] # Descending order
    
    # Get column names for the top 2 features (for Dependence Plots)
    # We use X_display so we get the Publication Names directly
    top_feat_1 = X_display.columns[top_indices[0]]
    top_feat_2 = X_display.columns[top_indices[1]]
    
    print(f"🔹 Top Feature 1: {top_feat_1}")
    print(f"🔹 Top Feature 2: {top_feat_2}")

    # 4. Create Layout
    fig = plt.figure(figsize=(24, 20), dpi=300)
    # Grid: Top row (A) spans full width. Bottom row split into (B) and (C).
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.25)
    
    # -------------------------------------------------------
    # PANEL (A): BEESWARM
    # -------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    
    # SHAP's summary_plot usually creates its own figure. 
    # We must pass 'show=False' and use the current axis.
    plt.sca(ax_a)
    
    shap.summary_plot(
        shap_values, 
        X_display, 
        plot_type="dot", 
        max_display=10, 
        show=False, 
        sort=True,
        plot_size=None # Let matplotlib handle size
    )
    
    # Customizing SHAP's default output to match Nature style
    ax_a.set_title("(A) Global Feature Importance", fontsize=title_size, weight='bold', pad=15, loc='left')
    ax_a.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=label_size)
    ax_a.tick_params(axis='x', labelsize=tick_size)
    ax_a.tick_params(axis='y', labelsize=tick_size)
    
    # Fix the font of the Y-axis feature labels (SHAP sometimes hardcodes these)
    for label in ax_a.get_yticklabels():
        label.set_fontname(font_family)
        label.set_fontsize(tick_size)

    # -------------------------------------------------------
    # PANEL (B): DEPENDENCE PLOT 1 (Top Feature)
    # -------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    plt.sca(ax_b) # Set current axis
    
    shap.dependence_plot(
        top_indices[0], # Index of feature
        shap_values, 
        X_display, 
        display_features=X_display,
        interaction_index='auto', # Let SHAP find the color feature automatically
        show=False,
        ax=ax_b,
        alpha=0.7,
        x_jitter=0.1
    )
    
    ax_b.set_title(f"(B) Dependence: {top_feat_1}", fontsize=title_size, weight='bold', pad=15, loc='left')
    ax_b.set_ylabel("SHAP Value", fontsize=label_size)
    ax_b.set_xlabel(top_feat_1, fontsize=label_size)
    ax_b.tick_params(labelsize=tick_size)
    ax_b.grid(True, linestyle=':', alpha=0.4)

    # -------------------------------------------------------
    # PANEL (C): DEPENDENCE PLOT 2 (2nd Top Feature)
    # -------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 1])
    plt.sca(ax_c)
    
    shap.dependence_plot(
        top_indices[1], 
        shap_values, 
        X_display, 
        display_features=X_display,
        interaction_index='auto',
        show=False,
        ax=ax_c,
        alpha=0.7,
        x_jitter=0.1
    )
    
    ax_c.set_title(f"(C) Dependence: {top_feat_2}", fontsize=title_size, weight='bold', pad=15, loc='left')
    ax_c.set_ylabel("SHAP Value", fontsize=label_size) # Or remove to save space if redundant
    ax_c.set_xlabel(top_feat_2, fontsize=label_size)
    ax_c.tick_params(labelsize=tick_size)
    ax_c.grid(True, linestyle=':', alpha=0.4)

    # -------------------------------------------------------
    # SAVE & SHOW
    # -------------------------------------------------------
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=IMAGE_DPI)
        print(f"✅ Figure saved to: {save_path}")
        
    plt.show()

def get_custom_shap_cmap(color_start, color_end):
    colors = [color_start, color_end] 
    return LinearSegmentedColormap.from_list("CustomSHAP", colors, N=256)

def plot_shap_aligned_custom(
    shap_values,
    X_df,
    mapping_dict,
    save_path=None,
    # --- Font Sizes ---
    font_title=26,
    font_label=22,
    font_tick=18,
    # --- Colorbar Font Control ---
    font_cbar_label=20,  
    font_cbar_tick=16,   
    # --- Layout ---
    margin_left=0.25,
    margin_right=0.95,
    split_y=0.50,
    wspace_bottom=0.3
):
    """
    Generates a perfectly aligned SHAP figure with customized Colorbar fonts and Custom Colormaps.
    """

    # Rename Data
    X_display = X_df.rename(columns=mapping_dict)
    
    # Identify Top Features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1]
    top_feat_1 = X_display.columns[top_indices[0]]
    top_feat_2 = X_display.columns[top_indices[1]]

    fig = plt.figure(figsize=(24, 15), dpi=300)
    
    # --- HELPER: Fix SHAP's Hardcoded Colorbars ---
    def style_last_colorbar(fig, tick_size, label_size):
        cbar_ax = fig.axes[-1] 
        cbar_ax.tick_params(labelsize=tick_size, colors='black', width=1, length=4)
        
        # Style the label (Interaction Feature Name)
        current_label = cbar_ax.get_ylabel()
        cbar_ax.set_ylabel(current_label, fontsize=label_size, fontname='sans-serif', labelpad=10)
        
        for spine in cbar_ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1)

    # Grids
    gs_top = fig.add_gridspec(nrows=1, ncols=1, left=margin_left, right=margin_right, top=0.92, bottom=split_y + 0.08)
    gs_bot = fig.add_gridspec(nrows=1, ncols=2, left=margin_left, right=margin_right, top=split_y - 0.05, bottom=0.08, wspace=wspace_bottom)

    # -------------------------------------------------------
    # PANEL (C): BEESWARM 
    # -------------------------------------------------------
    ax_a = fig.add_subplot(gs_top[0])
    plt.sca(ax_a)
    n_axes_before = len(fig.axes)
    
    # Re-enabled the cmap argument passing our custom Alpha/Beta colors
    shap.summary_plot(
        shap_values, X_display, plot_type="dot", max_display=10, 
        show=False, sort=True, cmap=custom_shap_cmap, plot_size=None 
    )
    
    if len(fig.axes) > n_axes_before:
        style_last_colorbar(fig, font_cbar_tick, font_cbar_label)

    # Styling
    ax_a.set_title("C. Global Feature Importance", fontsize=font_title, weight='bold', pad=20, loc='left')
    ax_a.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=font_label, labelpad=10)
    ax_a.tick_params(axis='both', labelsize=font_tick)
    
    # Fix Y-axis font (Feature Names)
    for label in ax_a.get_yticklabels():
        label.set_fontsize(font_tick)
        label.set_color('black')

    # -------------------------------------------------------
    # PANEL (D): TOP FEATURE DEPENDENCE 
    # -------------------------------------------------------
    ax_b = fig.add_subplot(gs_bot[0])
    plt.sca(ax_b)
    n_axes_before = len(fig.axes)
    
    # Passed custom_shap_cmap here
    shap.dependence_plot(
        top_indices[0], shap_values, X_display, display_features=X_display,
        interaction_index='auto', show=False, ax=ax_b, alpha=0.7, x_jitter=0.1, cmap=custom_shap_cmap
    )
    
    if len(fig.axes) > n_axes_before:
        style_last_colorbar(fig, font_cbar_tick, font_cbar_label)

    ax_b.set_title(f"D. Dependence: {top_feat_1}", fontsize=font_title, weight='bold', pad=20, loc='left')
    ax_b.set_ylabel("SHAP Value", fontsize=font_label)
    ax_b.set_xlabel(top_feat_1, fontsize=font_label)
    ax_b.tick_params(labelsize=font_tick)
    ax_b.grid(True, linestyle=':', alpha=0.4)

    # -------------------------------------------------------
    # PANEL (E): 2ND FEATURE DEPENDENCE
    # -------------------------------------------------------
    ax_c = fig.add_subplot(gs_bot[1])
    plt.sca(ax_c)
    n_axes_before = len(fig.axes)
    
    # Passed custom_shap_cmap here
    shap.dependence_plot(
        top_indices[1], shap_values, X_display, display_features=X_display,
        interaction_index='auto', show=False, ax=ax_c, alpha=0.7, x_jitter=0.1, cmap=custom_shap_cmap
    )
    
    if len(fig.axes) > n_axes_before:
        style_last_colorbar(fig, font_cbar_tick, font_cbar_label)

    ax_c.set_title(f"E. Dependence: {top_feat_2}", fontsize=font_title, weight='bold', pad=20, loc='left')
    ax_c.set_ylabel("SHAP Value", fontsize=font_label)
    ax_c.set_xlabel(top_feat_2, fontsize=font_label)
    ax_c.tick_params(labelsize=font_tick)
    ax_c.grid(True, linestyle=':', alpha=0.4)

    # -------------------------------------------------------
    # SAVE
    # -------------------------------------------------------
    if save_path:
        plt.savefig(save_path, dpi=IMAGE_DPI, bbox_inches='tight') 
        print(f"✅ Figure saved to: {save_path}")
        
    plt.show()

def save_individual_dependence_plots(
    model_to_analyze,
    shap_values,
    X_df,
    mapping_dict,
    save_dir,
    top_n=5,
    cmap=custom_shap_cmap, 
    # --- TEXT SIZES ---
    font_title=28,
    font_label=24,
    font_tick=20,
    font_cbar_label=22,
    font_cbar_tick=18
):
    
    set_nature_style()
    """
    Loops through top N features, saves perfcet square plots with custom colormap,
    and ensures the colorbar does not intersect big axis labels.
    """
    
    # 1. Rename Data
    X_display = X_df.rename(columns=mapping_dict)
    
    # 2. Identify Top Features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1][:top_n]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"--- Generating {top_n} Individual Dependence Plots ---")
    print(f"--- Using Custom Colormap: Orange to Blue ---")

    for rank, feat_idx in enumerate(top_indices):
        feature_name = X_display.columns[feat_idx]
        print(f"Processing {rank+1}/{top_n}: {feature_name}...")

        # Initialize individual figure (slightly wider to accommodate cbar spacing)
        fig, ax = plt.subplots(figsize=(11, 10), dpi=300)
        
        # 3. Plot WITH CUSTOM CMAP
        shap.dependence_plot(
            feat_idx, shap_values, X_display, display_features=X_display,
            interaction_index='auto', show=False, ax=ax, alpha=0.8, x_jitter=0.1,
            cmap=cmap # <--- Applying your custom colormap here
        )

        # 4. Handle Colorbar (Delete SHAP's default)
        interaction_label = ""
        if len(fig.axes) > 1:
            bad_cbar_ax = fig.axes[-1]
            interaction_label = bad_cbar_ax.get_ylabel()
            fig.delaxes(bad_cbar_ax)

        # 5. Force Square Aspect Ratio
        ax.set_box_aspect(1)

        # 6. Re-attach Colorbar with MORE PADDING
        if interaction_label and ax.collections:
            sc = ax.collections[0]
            divider = make_axes_locatable(ax)
            
            # --- THE FIX FOR INTERSECTION IS HERE ---
            # Increased pad from 0.15 to 0.5 to push cbar further right
            cax = divider.append_axes("right", size="5%", pad=0.5) 
            
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label(interaction_label, fontsize=font_cbar_label, labelpad=20)
            cbar.ax.tick_params(labelsize=font_cbar_tick)
            cbar.outline.set_edgecolor('black')
            cbar.outline.set_linewidth(1)

        # 7. Styling
        safe_filename = "".join([c if c.isalnum() else "_" for c in feature_name])
        ax.set_title(f"Dependence: {feature_name}", fontsize=font_title, weight='bold', pad=25)
        ax.set_ylabel("SHAP Value", fontsize=font_label, labelpad=15)
        ax.set_xlabel(feature_name, fontsize=font_label, labelpad=15)
        ax.tick_params(labelsize=font_tick)
        ax.grid(True, linestyle=':', alpha=0.4)

        # 8. Save
        save_path = os.path.join(save_dir, f"Dependence_{rank+1}_{safe_filename}_{model_to_analyze}.png")
        plt.savefig(save_path, dpi=IMAGE_DPI, bbox_inches='tight')
        plt.close(fig)

    print(f"✅ All plots saved to: {save_dir}")

def plot_ultimate_interpretability_figure(
    # --- Permutation & PDP Data ---
    perm_summary_path,
    perm_raw_path,
    pdp_summary_path,
    custom_pdp_features,
    # --- SHAP Data ---
    shap_values,
    X_shap_df,
    # --- Mappings & Setup ---
    mapping_dict,
    custom_shap_cmap=custom_shap_cmap, 
    save_path=None,
    top_n_perm=10,
    box_ar=1.5,
    # --- Layout Controls ---
    midpoint_y=0.50,      # <--- NEW: The vertical center of the figure (0.0 to 1.0)
    vertical_gap=0.10,    # <--- NEW: The total empty space between top and bottom halves
    # --- Font Controls ---
    font_title=26,
    font_label=22,
    font_tick=18,
    font_cbar_label=20,  
    font_cbar_tick=16    
):
    # ==========================================================================
    # 0. DATA PREPARATION
    # ==========================================================================ù

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


    PUB_TO_TECH = {v: k for k, v in mapping_dict.items()}
    
    perm_path, raw_path, pdp_path = Path(perm_summary_path), Path(perm_raw_path), Path(pdp_summary_path)
    if not perm_path.exists() or not pdp_path.exists():
        print(f"❌ Error: Permutation or PDP data files missing.")
        return

    perm_df = pd.read_csv(perm_path, index_col=0)
    raw_df = pd.read_csv(raw_path, index_col=0) if raw_path.exists() else None
    pdp_df = pd.read_csv(pdp_path)

    df = perm_df.copy()
    if 'mean_importance' not in df.columns:
        if 'mean_importance' in (df.index.name or ''): df = df.reset_index()
        else:
            cols = df.columns.tolist()
            rename_dict = {}
            if len(cols) >= 2: rename_dict[cols[0]] = 'Feature'; rename_dict[cols[1]] = 'mean_importance'
            if len(cols) >= 3: rename_dict[cols[2]] = 'std_importance'
            df = df.rename(columns=rename_dict)
            if 'Feature' in df.columns: df.set_index('Feature', inplace=True)

    df = df.sort_values(by='mean_importance', ascending=True).tail(top_n_perm)
    
    is_pub_name = df.index[0] in mapping_dict.values()
    if is_pub_name:
        df['DisplayName'] = df.index
        df['TechName'] = [PUB_TO_TECH.get(x, x) for x in df.index]
    else:
        df['DisplayName'] = [mapping_dict.get(x, x) for x in df.index]
        df['TechName'] = df.index
    top_features_ordered = df.index.tolist()[::-1]

    X_display = X_shap_df.rename(columns=mapping_dict)
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1]
    top_shap_feat_1 = X_display.columns[top_indices[0]]
    top_shap_feat_2 = X_display.columns[top_indices[1]]

    # ==========================================================================
    # 1. FIGURE LAYOUT
    # ==========================================================================
    fig = plt.figure(figsize=(32, 24), dpi=300)
    
    # Calculate boundaries based on the gap
    top_section_bottom = midpoint_y + (vertical_gap / 2)
    bottom_section_top = midpoint_y - (vertical_gap / 2)
    
    # Quadrant A: Top Left (Permutation)
    gs_perm = fig.add_gridspec(1, 1, left=0.05, right=0.45, top=0.92, bottom=top_section_bottom)
    # Quadrant B: Top Right (PDP Grid)
    gs_pdp = fig.add_gridspec(2, 3, left=0.55, right=0.95, top=0.92, bottom=top_section_bottom, wspace=0.35, hspace=0.4)
    
    # Quadrant C: Bottom Left (SHAP Beeswarm)
    gs_shap = fig.add_gridspec(1, 1, left=0.05, right=0.45, top=bottom_section_top, bottom=0.08)
    
    # Quadrant D: Bottom Right (SHAP Dependence)
    gs_dep = fig.add_gridspec(1, 2, left=0.55, right=0.95, top=bottom_section_top, bottom=0.08, wspace=0.45)

    def style_last_colorbar(fig, tick_size, label_size):
        cbar_ax = fig.axes[-1] 
        cbar_ax.tick_params(labelsize=tick_size, colors='black', width=1.5, length=6)
        current_label = cbar_ax.get_ylabel()
        cbar_ax.set_ylabel(current_label, fontsize=label_size, fontname='serif', labelpad=15, weight='bold')
        for spine in cbar_ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)

    # ==========================================================================
    # QUADRANT A: PERMUTATION IMPORTANCE
    # ==========================================================================
    ax_perm = fig.add_subplot(gs_perm[0])
    sns.barplot(x='mean_importance', y='DisplayName', data=df, order=top_features_ordered, 
                color='skyblue', edgecolor='black', linewidth=1.5, ax=ax_perm, zorder=2)

    if 'std_importance' in df.columns:
        df_sorted = df.reindex(top_features_ordered)
        ax_perm.errorbar(x=df_sorted['mean_importance'], y=np.arange(len(df_sorted)), 
                         xerr=df_sorted['std_importance'], fmt='none', c='black', capsize=6, linewidth=2, zorder=4)
        
    if raw_df is not None:
        raw_cols, target_tech, target_disp = set(raw_df.columns), set(df['TechName']), set(df['DisplayName'])
        melted = None
        valid_cols_disp = raw_cols.intersection(target_disp)
        if len(valid_cols_disp) > 0:
            melted = raw_df[list(valid_cols_disp)].melt(var_name='DisplayName', value_name='Importance')
        elif len(raw_cols.intersection(target_tech)) > 0:
            valid_cols_tech = list(raw_cols.intersection(target_tech))
            melted = raw_df[valid_cols_tech].melt(var_name='TechName', value_name='Importance')
            tech_map = dict(zip(df['TechName'], df['DisplayName']))
            melted['DisplayName'] = melted['TechName'].map(tech_map)
            
        if melted is not None:
            sns.stripplot(x='Importance', y='DisplayName', data=melted, order=top_features_ordered,
                          color='#C44E52', size=6, jitter=0.25, ax=ax_perm, zorder=10, linewidth=0.5, edgecolor='black')

    ax_perm.spines['left'].set_linewidth(2)
    ax_perm.spines['bottom'].set_linewidth(2)
    ax_perm.spines['top'].set_visible(False)
    ax_perm.spines['right'].set_visible(False)
    ax_perm.tick_params(axis='both', which='major', width=2, length=8, labelsize=font_tick)
    ax_perm.set_xlabel('Relative Mean Importance Score', fontsize=font_label, fontweight='bold', labelpad=15)
    ax_perm.set_ylabel('')
    ax_perm.set_title('(A) Permutation Feature Importance', fontsize=font_title, fontweight='bold', loc='left', pad=20)

    # ==========================================================================
    # QUADRANT B: PARTIAL DEPENDENCE PLOTS (PDP)
    # ==========================================================================
    fig.text(0.55, 0.94, "(B) Partial Dependence Plots (PDP)", fontsize=font_title, fontweight='bold')
    
    pdp_tech_names = custom_pdp_features if custom_pdp_features else df['TechName'].tolist()[::-1][:6]
    handles = []
    
    for i, tech_feat in enumerate(pdp_tech_names):
        row, col = i // 3, i % 3
        if row >= 2: break 
            
        ax = fig.add_subplot(gs_pdp[row, col])
        subset = pdp_df[pdp_df['feature'] == tech_feat]
        
        if not subset.empty:
            subset = subset.sort_values('grid_value')
            l, = ax.plot(subset['grid_value'], subset['mean'], color='black', linewidth=3)
            f = ax.fill_between(subset['grid_value'], subset['mean'] - subset['std'], subset['mean'] + subset['std'],
                                color='skyblue', alpha=0.5)
            if i == 0: handles = [l, f]
            
        pub_name = mapping_dict.get(tech_feat, tech_feat)
        ax.set_xlabel(pub_name, fontweight='bold', fontsize=font_tick)
        if col == 0: ax.set_ylabel("Partial Dependence", fontsize=font_label, fontweight='bold', labelpad=10)
            
        ax.tick_params(axis='both', which='major', bottom=True, left=True, width=1.5, length=6, labelsize=font_tick-2)
        ax.spines['left'].set_linewidth(1.5); ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=':', alpha=0.6)

    if handles:
        fig.legend(handles, ['Mean Effect', '±1 Std. Dev.'], loc='upper center', 
                   bbox_to_anchor=(0.75, 0.535), ncol=2, frameon=False, fontsize=font_tick)

    # ==========================================================================
    # QUADRANT C: SHAP BEESWARM
    # ==========================================================================
    ax_shap = fig.add_subplot(gs_shap[0])
    plt.sca(ax_shap)
    n_axes_before = len(fig.axes)
    
    shap.summary_plot(shap_values, X_display, plot_type="dot", max_display=top_n_perm, 
                      show=False, sort=True, cmap=custom_shap_cmap, plot_size=None)
    
    if len(fig.axes) > n_axes_before: style_last_colorbar(fig, font_cbar_tick, font_cbar_label)

    ax_shap.set_title("(C) SHAP Global Importance", fontsize=font_title, weight='bold', pad=20, loc='left')
    ax_shap.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=font_label, weight='bold', labelpad=15)
    ax_shap.tick_params(axis='both', labelsize=font_tick, width=2, length=8)
    for label in ax_shap.get_yticklabels():
        label.set_fontsize(font_tick)
        label.set_color('black')

    # ==========================================================================
    # QUADRANT D: SHAP DEPENDENCE PLOTS
    # ==========================================================================
    
    # [FIXED] Title y-coordinate dynamically rests inside the gap we just created
    title_d_y = bottom_section_top + (vertical_gap * 0.4)
    fig.text(0.55, title_d_y, "(D) SHAP-Dependence Plot", fontsize=font_title, fontweight='bold', ha='left', va='center')

    # 2. Subplot 1
    ax_dep1 = fig.add_subplot(gs_dep[0])
    plt.sca(ax_dep1)
    n_axes_before = len(fig.axes)
    shap.dependence_plot(top_indices[0], shap_values, X_display, display_features=X_display,
                         interaction_index='auto', show=False, ax=ax_dep1, alpha=0.7, x_jitter=0.1, cmap=custom_shap_cmap)
    if len(fig.axes) > n_axes_before: style_last_colorbar(fig, font_cbar_tick, font_cbar_label)
    
    ax_dep1.set_title(top_shap_feat_1, fontsize=font_title - 4, weight='bold', pad=10, loc='center')
    ax_dep1.set_ylabel("SHAP Value", fontsize=font_label, weight='bold')
    ax_dep1.set_xlabel(top_shap_feat_1, fontsize=font_label, weight='bold')
    ax_dep1.tick_params(labelsize=font_tick, width=2, length=8)
    ax_dep1.grid(True, linestyle=':', alpha=0.4)
    ax_dep1.set_box_aspect(box_ar) 

    # 3. Subplot 2
    ax_dep2 = fig.add_subplot(gs_dep[1])
    plt.sca(ax_dep2)
    n_axes_before = len(fig.axes)
    shap.dependence_plot(top_indices[1], shap_values, X_display, display_features=X_display,
                         interaction_index='auto', show=False, ax=ax_dep2, alpha=0.7, x_jitter=0.1, cmap=custom_shap_cmap)
    if len(fig.axes) > n_axes_before: style_last_colorbar(fig, font_cbar_tick, font_cbar_label)

    ax_dep2.set_title(top_shap_feat_2, fontsize=font_title - 4, weight='bold', pad=10, loc='center')
    ax_dep2.set_ylabel("SHAP Value", fontsize=font_label, weight='bold')
    ax_dep2.set_xlabel(top_shap_feat_2, fontsize=font_label, weight='bold')
    ax_dep2.tick_params(labelsize=font_tick, width=2, length=8)
    ax_dep2.grid(True, linestyle=':', alpha=0.4)
    ax_dep2.set_box_aspect(box_ar)

    # ==========================================================================
    # SAVE & SHOW
    # ==========================================================================
    if save_path:
        plt.savefig(save_path, dpi=IMAGE_DPI, bbox_inches='tight') 
        print(f"✅ Ultimate Figure saved to: {save_path}")
        
    plt.show()






