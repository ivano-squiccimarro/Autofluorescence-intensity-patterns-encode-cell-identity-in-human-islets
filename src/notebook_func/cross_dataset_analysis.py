
import joblib
import numpy        as np
import pandas       as pd
import seaborn      as sns

import matplotlib.pyplot        as plt
import matplotlib.lines         as mlines
import matplotlib.ticker        as ticker
import matplotlib.gridspec      as gridspec

from scipy.stats    import wilcoxon

# Local Imports
from src.feature_extraction_functions import extract_feats
from src.notebook_func.results_retrieval import compute_all_metrics
from config.presentation_config import set_nature_style, CLASS_COLORS, IMAGE_DPI


# ____________________________________________________________________________________________
# Relative Paths and Order of Appearance

ORDER_LIST              = ["Dataset++", "Dataset_SMOTE", "Dataset"]
SOURCE_PALETTE          = "viridis"

DATASET_A_FILE_REL_PATH = "Datasets/Training_Test_Splits/Global_Data/train/global_training_set.pkl"
DATASET_B_FILE_REL_PATH = "Datasets/Training_Test_Splits/Global_Data/test/test_set.pkl"
SCALER_REL_PATH         = "Datasets/Training_Test_Splits/Global_Data/Global_Scalers/global_scaler.pkl"
FEATURES_LIST_REL_PATH  = "Datasets/Training_Test_Splits/features_to_keep.pkl"
MODEL_REL_PATH          = "Results/Final_Results/Test_Results/LGBM_final_model.pkl"

#_____________________________________________________________________
# Intra Dataset Comparison: 

def apply_journal_spines(ax, tick_fontsize, order_list):
    """Clean journal spines with a strict tick enforcement."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    ax.tick_params(axis='both', labelsize=tick_fontsize, direction='out', length=5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(order_list))))
    ax.set_xticklabels(order_list, rotation=15)
    ax.xaxis.set_minor_locator(ticker.NullLocator())

def generate_comparison_figure(df, 
                               boxplot_width=0.4, 
                               table_font_size=20, 
                               title_fontsize=22, 
                               header_fontsize=18,  # Subtitle font size
                               ylabel_fontsize=16,  # Y-axis label font size
                               tick_fontsize=17,
                               show=True):
    
    # Ensure a clean aesthetic
    sns.set_style("ticks")
    set_nature_style()
    
    df_test = df[df['stress_level'] == 0].copy()
    df_stress = df[df['stress_level'] == 3].copy()
    
    fig = plt.figure(figsize=(24, 16))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    
    fig.suptitle("LGBM Intra-Dataset Performances Analysis: Test vs Stress Test Level 3", 
                 fontsize=title_fontsize, weight='bold', y=0.98)

    for row_idx, (data_sub, row_title) in enumerate([(df_test, "TEST SET (Lvl 0)"), 
                                                     (df_stress, "STRESS SET (Lvl 3)")]):
        
        # Adjusting width ratios to give the table (index 2) more space
        gs_row = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[row_idx], 
                                                 width_ratios=[0.7, 0.7, 1.4], wspace=0.25)
        
        # --- Plot A: General ---
        ax_gen = fig.add_subplot(gs_row[0])
        sns.boxplot(data=data_sub, x='data_source', y='precision', hue='data_source', 
                    order=ORDER_LIST, palette=SOURCE_PALETTE, width=boxplot_width, 
                    ax=ax_gen, showfliers=False, legend=False)
        
        ax_gen.set_title(f"General Precision", fontsize=header_fontsize, weight='bold', pad=15)
        ax_gen.set_ylabel("Precision Score", fontsize=ylabel_fontsize)
        ax_gen.set_xlabel("") 
        apply_journal_spines(ax_gen, tick_fontsize, ORDER_LIST)
        
        # --- Plot B: Per-Class ---
        ax_cls = fig.add_subplot(gs_row[1])
        melted = data_sub.melt(id_vars=['data_source'], value_vars=['precision_0', 'precision_1'], var_name='C')
        melted['Class'] = melted['C'].map({'precision_0': 'Alpha', 'precision_1': 'Beta'})
        
        sns.boxplot(data=melted, x='data_source', y='value', hue='Class', order=ORDER_LIST,
                    palette={"Alpha": CLASS_COLORS['Alpha'], "Beta": CLASS_COLORS['Beta']}, width=boxplot_width+0.1, 
                    ax=ax_cls, showfliers=False)
        
        ax_cls.set_title(f"Per-Class Precision", fontsize=header_fontsize, weight='bold', pad=15)
        ax_cls.set_ylabel("Precision Score", fontsize=ylabel_fontsize)
        ax_cls.set_xlabel("") 
        ax_cls.get_legend().remove()
        apply_journal_spines(ax_cls, tick_fontsize, ORDER_LIST)

        # --- Table C: Summary Table ---
        ax_tab = fig.add_subplot(gs_row[2])
        stats = data_sub.groupby('data_source')[['precision', 'precision_0', 'precision_1']].agg(['mean', 'std'])
        
        cell_text = []
        for s in ORDER_LIST:
            row = [s]
            row.append(f"{stats.loc[s, ('precision', 'mean')]:.2f} ± {stats.loc[s, ('precision', 'std')]:.2f}")
            row.append(f"{stats.loc[s, ('precision_0', 'mean')]:.2f} ± {stats.loc[s, ('precision_0', 'std')]:.2f}")
            row.append(f"{stats.loc[s, ('precision_1', 'mean')]:.2f} ± {stats.loc[s, ('precision_1', 'std')]:.2f}")
            cell_text.append(row)
            
        col_labels = ['Data Source', 'General', 'Alpha (α)', 'Beta (β)']
        source_col_widths = [0.34, 0.22, 0.22, 0.22]
        
        generate_journal_style_table(ax_tab, cell_text, col_labels, table_font_size, source_col_widths)
        ax_tab.set_title(f"Summary Statistics: {row_title}", fontsize=header_fontsize, weight='bold', pad=25)

    # Global Legend
    h_alpha = mlines.Line2D([], [], color="#1f77b4", marker='s', ls='', label='Alpha')
    h_beta = mlines.Line2D([], [], color="#ff7f0e", marker='s', ls='', label='Beta')
    fig.legend(handles=[h_alpha, h_beta], loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, 0.02), fontsize=header_fontsize, frameon=False)

    if show: plt.show()

def generate_journal_style_table(ax, cell_text, col_labels, font_size, col_widths):
    ax.axis('off')
    n_rows = len(cell_text)
    table = ax.table(cellText=cell_text, colLabels=col_labels, colWidths=col_widths, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white') 
        if row == 0: cell.set_text_props(weight='bold')
        if col == 0: cell.set_text_props(ha='left')
            
    ax.plot([0, 1], [1, 1], color='black', lw=2, transform=ax.transAxes, clip_on=False)
    h_y = 1 - (1 / (n_rows + 1))
    ax.plot([0, 1], [h_y, h_y], color='black', lw=1, transform=ax.transAxes)
    ax.plot([0, 1], [0, 0], color='black', lw=2, transform=ax.transAxes)
    return table

def generate_intra_dataset_comparison(df, order_list, source_palette, save_path=None, show=False):
    """Generates the Test vs Stress boxplot figure across datasets."""
    set_nature_style()
    sns.set_style("ticks")
    
    df_test = df[df['stress_level'] == 0].copy()
    df_stress = df[df['stress_level'] == 3].copy()
    
    fig = plt.figure(figsize=(24, 16))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    fig.suptitle("LGBM Intra-Dataset Performances Analysis: Test vs Stress Test Level 3", fontsize=22, weight='bold', y=0.98)

    for row_idx, (data_sub, row_title) in enumerate([(df_test, "TEST SET (Lvl 0)"), (df_stress, "STRESS SET (Lvl 3)")]):
        gs_row = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[row_idx], width_ratios=[0.7, 0.7, 1.4], wspace=0.25)
        
        # Plot A: General
        ax_gen = fig.add_subplot(gs_row[0])
        sns.boxplot(data=data_sub, x='data_source', y='precision', hue='data_source', order=order_list, palette=source_palette, width=0.4, ax=ax_gen, showfliers=False, legend=False)
        ax_gen.set_title("General Precision", fontsize=18, weight='bold', pad=15)
        ax_gen.set_ylabel("Precision Score", fontsize=16)
        ax_gen.set_xlabel("") 
        apply_journal_spines(ax_gen, 17, order_list)
        
        # Plot B: Per-Class
        ax_cls = fig.add_subplot(gs_row[1])
        melted = data_sub.melt(id_vars=['data_source'], value_vars=['precision_0', 'precision_1'], var_name='C')
        melted['Class'] = melted['C'].map({'precision_0': 'Alpha', 'precision_1': 'Beta'})
        sns.boxplot(data=melted, x='data_source', y='value', hue='Class', order=order_list, palette={"Alpha": CLASS_COLORS['Alpha'], "Beta": CLASS_COLORS['Beta']}, width=0.5, ax=ax_cls, showfliers=False)
        ax_cls.set_title("Per-Class Precision", fontsize=18, weight='bold', pad=15)
        ax_cls.set_ylabel("Precision Score", fontsize=16)
        ax_cls.set_xlabel("") 
        if ax_cls.get_legend(): ax_cls.get_legend().remove()
        apply_journal_spines(ax_cls, 17, order_list)

        # Table C
        ax_tab = fig.add_subplot(gs_row[2])
        stats = data_sub.groupby('data_source')[['precision', 'precision_0', 'precision_1']].agg(['mean', 'std'])
        cell_text = []
        for s in order_list:
            if s in stats.index:
                cell_text.append([
                    s,
                    f"{stats.loc[s, ('precision', 'mean')]:.2f} ± {stats.loc[s, ('precision', 'std')]:.2f}",
                    f"{stats.loc[s, ('precision_0', 'mean')]:.2f} ± {stats.loc[s, ('precision_0', 'std')]:.2f}",
                    f"{stats.loc[s, ('precision_1', 'mean')]:.2f} ± {stats.loc[s, ('precision_1', 'std')]:.2f}"
                ])
        generate_journal_style_table(ax_tab, cell_text, ['Data Source', 'General', 'Alpha (α)', 'Beta (β)'], 20, [0.34, 0.22, 0.22, 0.22])
        ax_tab.set_title(f"Summary Statistics: {row_title}", fontsize=18, weight='bold', pad=25)

    h_alpha = mlines.Line2D([], [], color=CLASS_COLORS['Alpha'], marker='s', ls='', label='Alpha')
    h_beta = mlines.Line2D([], [], color=CLASS_COLORS['Beta'], marker='s', ls='', label='Beta')
    fig.legend(handles=[h_alpha, h_beta], loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=18, frameon=False)

    if save_path:
        plt.savefig(save_path, dpi=IMAGE_DPI, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)

#_____________________________________________________________________
# Inter Dataset Comparison: 


def get_integer_indices(feature_list, all_columns):
    """Converts feature names to integer indices safely."""
    if len(feature_list) == 0: return []
    if isinstance(feature_list[0], (str, np.str_)):
        col_map = {name: i for i, name in enumerate(all_columns)}
        return [col_map[feat] for feat in feature_list if feat in col_map]
    return feature_list

def get_comparison_summary(results_df, name_a, name_b, threshold=0.01, only_significant=True):
    """
    Calculates Mean/STD and P-values.
    Args:
        threshold (float): The cutoff for significance (e.g., 0.01 or 0.05).
        only_significant (bool): If True, hides rows where p_value > threshold.
    """
    metrics_map = {
        'ACCURACY': 'Accuracy',
        'AUC': 'AUC',
        'F1_BETA': 'F1 Score (Beta)',
        'PRECISION_BETA': 'Precision (Beta)',
        'RECALL_BETA': 'Recall (Beta)',
        'F1_ALPHA': 'F1 Score (Alpha)',
        'PRECISION_ALPHA': 'Precision (Alpha)',
        'RECALL_ALPHA': 'Recall (Alpha)',
    }
    
    summary_rows = []
    
    for metric_suffix, metric_name in metrics_map.items():
        col_a = f'{name_a}_{metric_suffix}'
        col_b = f'{name_b}_{metric_suffix}'
        
        if col_a not in results_df.columns or col_b not in results_df.columns:
            continue
            
        # Stats
        mean_a, std_a = results_df[col_a].mean(), results_df[col_a].std()
        mean_b, std_b = results_df[col_b].mean(), results_df[col_b].std()
        
        str_a = f"{mean_a:.2f} ± {std_a:.2f}"
        str_b = f"{mean_b:.2f} ± {std_b:.2f}"
        
        # Wilcoxon Test
        p_str, sig = "-", "ns"
        p_val = 1.0
        
        try:
            valid_data = results_df[[col_a, col_b]].dropna()
            if len(valid_data) > 1:
                stat, p_val = wilcoxon(valid_data[col_a], valid_data[col_b])
                p_str = f"{p_val:.4f}"
                
                # Determine Stars based on standard scientific notation
                if p_val < 0.01: 
                    sig = "**"
                elif p_val < 0.05: 
                    sig = "*"
                else:
                    sig = "ns"
        except Exception:
            pass
            
        # FILTER LOGIC: 
        # If user wants strict significance (e.g. 0.01), we skip if p_val is higher than that.
        if only_significant:
            if p_val >= threshold: 
                continue
            
        summary_rows.append({
            'Metric': metric_name,
            f'{name_a}\n (Mean ± SD)': str_a,
            f'{name_b}\n (Mean ± SD)': str_b,
            'P-Value': p_str,
            'Sig.': sig
        })
        
    if not summary_rows:
        return pd.DataFrame() # Return empty if no results meet the threshold
        
    return pd.DataFrame(summary_rows)

def run_comparison_and_get_data(
        dataset_a_path, 
        dataset_b_path,
        dataset_a_file_rel_path,
        dataset_b_file_rel_path,
        p_value_threshold:float = 0.01, 
        verbosity: int = 0,
         
):
    # Dynamic Name Retrieval
    name_a = dataset_a_path
    name_b = dataset_b_path


    results_storage = {}
    print(f"🔄 Processing: {name_a} vs {name_b} ...")

    for seed in range(10):
        try:
            seed_dir = f"Results_{seed}"
            path_a_seed = dataset_a_path / seed_dir
            path_b_seed = dataset_b_path / seed_dir
            
            if not path_a_seed.exists() or not path_b_seed.exists():
                print(f"⚠️  Skipping Seed {seed}: Directory not found.")
                continue

            # 1. Load Dataframes
            df_a = joblib.load(path_a_seed / dataset_a_file_rel_path)
            df_b = joblib.load(path_b_seed / dataset_b_file_rel_path,)

            # 2. Identify Unique Indices (B minus A)
            unique_indices = set(df_b.index) - set(df_a.index)
            num_unique = len(unique_indices)
            if verbosity > 0:
                print(f"Seed {seed}: Found {num_unique} unique indices.")

            if num_unique == 0: continue
                
            df_b_unique = df_b.loc[list(unique_indices)]
            
            unique_features_df = extract_feats(image_df=df_b_unique, n_jobs=-1, batch_size=64, verbosity=0)
            unique_feats_only = unique_features_df.drop(columns=['label'])
            unique_labels = unique_features_df['label']

            label_counts = pd.Series(['alpha' if val==0 else 'beta' for val in unique_features_df['label']]).value_counts()
            if verbosity > 0:
                print(f"For seed {seed}:\n {label_counts}")

            all_cols = unique_feats_only.columns.tolist()

            # 4. Load Scalers & Features List
            # ---> FIX 1: Safely check if files exist before trying to load them
            if not (path_a_seed / SCALER_REL_PATH).exists() or not (path_b_seed / SCALER_REL_PATH).exists():
                print(f"⚠️  Skipping Seed {seed}: Missing scaler files (likely incomplete run).")
                continue
                
            scaler_a = joblib.load(path_a_seed / SCALER_REL_PATH)
            scaler_b = joblib.load(path_b_seed / SCALER_REL_PATH)
            
            feats_raw_a = joblib.load(path_a_seed / FEATURES_LIST_REL_PATH)
            feats_raw_b = joblib.load(path_b_seed / FEATURES_LIST_REL_PATH)

            feats_idx_a = get_integer_indices(feats_raw_a, all_cols)
            feats_idx_b = get_integer_indices(feats_raw_b, all_cols)

            # 5. Transform Data
            # Pipeline A
            try:
                scaled_full_a = scaler_a.transform(unique_feats_only.values)
                input_a = scaled_full_a[:, feats_idx_a]
            except ValueError:
                input_a = scaler_a.transform(unique_feats_only.values[:, feats_idx_a])

            # Pipeline B
            try:
                input_b = scaler_b.transform(unique_feats_only.values[:, feats_idx_b])
            except ValueError:
                scaled_full_b = scaler_b.transform(unique_feats_only.values)
                input_b = scaled_full_b[:, feats_idx_b]

            # 6. Predict
            # ---> FIX 2: Check models exist, and extract only the positive class probability [:, 1]
            if not (path_a_seed / MODEL_REL_PATH).exists() or not (path_b_seed / MODEL_REL_PATH).exists():
                print(f"⚠️  Skipping Seed {seed}: Missing model files.")
                continue

            model_a = joblib.load(path_a_seed / MODEL_REL_PATH)
            model_b = joblib.load(path_b_seed / MODEL_REL_PATH)

            probas_a = model_a.predict_proba(input_a)[:, 1]
            probas_b = model_b.predict_proba(input_b)[:, 1]
            
            # 7. Metrics
            metrics_a = compute_all_metrics(unique_labels, probas_a)
            metrics_b = compute_all_metrics(unique_labels, probas_b)
            
            results_storage[seed] = {'metrics_a': metrics_a, 'metrics_b': metrics_b}
            if verbosity > 0:
                print(f"   ✅ Seed {seed} calculated. {name_a} Acc: {metrics_a.get('accuracy'):.3f}, {name_b} Acc: {metrics_b.get('accuracy'):.3f}")

        except Exception as e:
            print(f"❌ Seed {seed} Error: {e}")
    
    # ----------------------------------
    metrics_list = []
    metric_map = {
        'accuracy': 'ACCURACY',
        'AUC': 'AUC',
        'recall': 'RECALL',
        'precision': 'PRECISION',
        'precision_1': 'PRECISION_BETA',  
        'precision_0': 'PRECISION_ALPHA',  
        'recall_1': 'RECALL_BETA',       
        'recall_0': 'RECALL_ALPHA',         
        'f1_1': 'F1_BETA',
        'f1_0': 'F1_ALPHA'
    }

    for seed, data in results_storage.items():
        row = {'Seed': seed}
        for key, suffix in metric_map.items():
            row[f'{name_a}_{suffix}'] = data['metrics_a'].get(key, np.nan)
            row[f'{name_b}_{suffix}'] = data['metrics_b'].get(key, np.nan)
        metrics_list.append(row)

    if not metrics_list:
        # Fallback if list empty
        return name_a, name_b, pd.DataFrame() 

    results_df = pd.DataFrame(metrics_list)

    # Calculate the Summary Table
    summary_df = get_comparison_summary(results_df, name_a, name_b, p_value_threshold, only_significant=True)
    
    return name_a, name_b, summary_df

def get_comparison_summary(results_df, name_a, name_b, threshold=0.01, only_significant=True):
    """Refactored to eliminate the Sig column and handle empty cases gracefully."""
    metrics_map = {
        'ACCURACY': 'Accuracy',
        'AUC': 'AUC',
        'F1_BETA': 'F1 Score (Beta)',
        'PRECISION_BETA': 'Precision (Beta)',
        'RECALL_BETA': 'Recall (Beta)',
        'F1_ALPHA': 'F1 Score (Alpha)',
        'PRECISION_ALPHA': 'Precision (Alpha)',
        'RECALL_ALPHA': 'Recall (Alpha)',
    }
    
    summary_rows = []
    for metric_suffix, metric_name in metrics_map.items():
        col_a = f'{name_a}_{metric_suffix}'
        col_b = f'{name_b}_{metric_suffix}'
        
        if col_a not in results_df.columns or col_b not in results_df.columns:
            continue
            
        mean_a, std_a = results_df[col_a].mean(), results_df[col_a].std()
        mean_b, std_b = results_df[col_b].mean(), results_df[col_b].std()
        
        str_a = f"{mean_a:.2f} ± {std_a:.2f}"
        str_b = f"{mean_b:.2f} ± {std_b:.2f}"
        
        p_str = "-"
        p_val = 1.0
        try:
            from scipy.stats import wilcoxon
            valid_data = results_df[[col_a, col_b]].dropna()
            if len(valid_data) > 1:
                stat, p_val = wilcoxon(valid_data[col_a], valid_data[col_b])
                p_str = f"{p_val:.4f}"
        except Exception:
            pass
            
        if only_significant and p_val >= threshold:
            continue
            
        summary_rows.append({
            'Performance Metric': metric_name, # Renamed for more space
            f'{name_a}\n(Mean ± SD)': str_a,
            f'{name_b}\n(Mean ± SD)': str_b,
            'P-Value': p_str
        })
        
    return pd.DataFrame(summary_rows)

def create_quadrant_figure(data_collection, 
                           sub_title_fontsize=32,
                           ax_title_fontsize=26,
                           table_fontsize=23,
                           filename="Final_Article_Comparison_Table.png"):
    """
    Plots 4 minimal tables in a 2x2 Grid with specific row-wise organization.
    Row 1: Dataset vs Dataset++
    Row 2: Dataset_SMOTE vs Dataset++
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(28, 18))
    # Adjust layout to give figure "breath"
    plt.subplots_adjust(wspace=0.3, hspace=0.4) 
    
    fig.suptitle("Cross-Dataset Generalizability Analysis (LGBM Model)", 
                 fontsize=sub_title_fontsize, weight='bold', y=0.98)

    # Expanded width for the first column [Metric, Source A, Source B, P-Value]
    custom_col_widths = [0.35, 0.25, 0.25, 0.15]

    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        if i >= len(data_collection): continue
            
        name_a, name_b, df_orig = data_collection[i]
        df = df_orig.copy()

        # 1. Clean data: Ensure Dataset++ is always the first data column
        cols = list(df.columns)
        if len(cols) >= 3:
            if "Dataset++" in cols[2] and "Dataset++" not in cols[1]:
                new_col_order = [cols[0], cols[2], cols[1]] + cols[3:]
                df = df[new_col_order]

        # 2. Format P-Values (no red text logic)
        for col in df.columns:
            if 'P-Value' in col:
                df[col] = df[col].apply(lambda x: f"{float(x):.3f}" if (isinstance(x, (float, int)) or (isinstance(x, str) and x != "-")) else x)

        # 3. Titles with Context
        test_context = "on Unique Dataset++ Images" if i % 2 == 0 else "on Unique Dataset Images"
        ax.set_title(f"Comparison {i+1}: {test_context}", 
                     fontsize=ax_title_fontsize, weight='bold', pad=30, loc='center')

        if df.empty:
            ax.text(0.5, 0.5, "No significant differences found", 
                    ha='center', va='center', fontsize=table_fontsize, style='italic')
            continue

        # 4. Draw Table
        mpl_table = ax.table(cellText=df.values, colLabels=df.columns, 
                             colWidths=custom_col_widths,
                             loc='center', cellLoc='center', bbox=[0, 0, 1, 0.85])
        
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(table_fontsize)

        # 5. Styling
        for (row, col), cell in mpl_table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_linewidth(0)
            
            if row == 0:
                cell.set_text_props(weight='bold')
                if col == 0: cell.set_text_props(ha='left')
            else:
                # Metric column left-aligned, others centered
                if col == 0: cell.set_text_props(ha='left')
                
                # Bolding significant P-values instead of using red
                if 'P-Value' in df.columns[col]:
                    try:
                        if float(cell.get_text().get_text()) < 0.05:
                            cell.set_text_props(weight='bold')
                    except: pass

        # 6. Minimalist Lines (Journal Style)
        ax.plot([0, 1], [0.85, 0.85], color='black', lw=2.5, transform=ax.transAxes)
        h_sep = 0.85 - (0.85 / (len(df) + 1))
        ax.plot([0, 1], [h_sep, h_sep], color='black', lw=1.5, transform=ax.transAxes)
        ax.plot([0, 1], [0, 0], color='black', lw=2.5, transform=ax.transAxes)

    plt.savefig(filename, dpi=IMAGE_DPI, bbox_inches='tight')
    plt.show()
