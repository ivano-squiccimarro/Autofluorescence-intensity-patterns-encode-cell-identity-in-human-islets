import gc
import re
import sys
import joblib
import tracemalloc

from   time       import perf_counter
from   typing     import Any, List, Union, Dict
from   pathlib    import Path

import pandas   as pd
import numpy    as np 

import seaborn              as sns
import matplotlib.pyplot    as plt
from   matplotlib.ticker    import FuncFormatter 
from   matplotlib.lines     import Line2D

from sklearn.metrics        import roc_auc_score
from sklearn.preprocessing  import MinMaxScaler

from scipy import stats 

from config.presentation_config         import IMAGE_DPI, set_nature_style
from src.dataset_augmentation_functions import create_random_augmented_dataset
from src.feature_extraction_functions   import extract_feats


MODEL_TO_ANALYZE    = 'LGBM'
COLORS              = {'Old Pipeline (Serial)': '#87CEEB', 
                       'New Engine (RealTime)': '#e74c3c'
                      }







# =======================================================================================================
# Scalability Analysis 
# =======================================================================================================
def _run_single_model_benchmark(
    image_df: pd.DataFrame, 
    selected_indices: List[str], 
    model: Any, 
    n_augmentations: int, 
    n_warmup: int, 
    n_repeats: int, 
    extraction_mode: str
) -> Dict[str, Union[float, List[float]]]:
    """
    Worker function: Benchmarks a single model pipeline.
    """
    if extraction_mode == 'serial':
        n_jobs_val = 1
    elif extraction_mode == 'parallel':
        n_jobs_val = -1
    else:
        raise ValueError(f"Invalid extraction_mode: '{extraction_mode}'.")

    # Only augment if requested
    if n_augmentations > 0:
        try:
            augmented_only_df = create_random_augmented_dataset(
                image_df, n_augmentations=n_augmentations, balance=False, random_state=42
            )
            benchmark_df = pd.concat([image_df, augmented_only_df], ignore_index=True)
        except NameError:
             benchmark_df = pd.concat([image_df] * (n_augmentations + 1), ignore_index=True)
    else:
        benchmark_df = image_df.copy()

    benchmark_size = len(benchmark_df)
    feat_times, pred_times, total_times = [], [], []

    for i in range(n_warmup + n_repeats):
        total_start_time = perf_counter()

        feat_start_time = perf_counter()
        feature_df = extract_feats(benchmark_df, n_jobs=n_jobs_val)
        feat_end_time = perf_counter()

        valid_cols = [c for c in selected_indices if c in feature_df.columns]
        feature_df = feature_df[valid_cols]
        X_test = feature_df 
        if 'label' in X_test.columns:
             X_test = X_test.drop(columns=['label'])
        
        pred_start_time = perf_counter()
        _ = model.predict(X_test)
        pred_end_time = perf_counter()
        
        total_end_time = perf_counter()

        if i >= n_warmup:
            feat_times.append(feat_end_time - feat_start_time)
            pred_times.append(pred_end_time - pred_start_time)
            total_times.append(total_end_time - total_start_time)

    if not total_times:
        raise ValueError("No timing data collected.")

    mean_total_time = np.mean(total_times)
    
    return {
        'throughput_ips': benchmark_size / mean_total_time,
        'total_latency_ms': (mean_total_time / benchmark_size) * 1000,
        'feat_time_s': np.mean(feat_times),
        'pred_time_s': np.mean(pred_times),
        'feat_times_raw': feat_times,
        'pred_times_raw': pred_times
    }

def analyze_model_generalization(
    image_df,
    all_models,
    selected_indices,
    save_dir,
    n_augmentations=1,
    n_warmup=5,
    n_repeats=10,
    mode='parallel'
):
    print(f"\n{'='*60}\nMODEL GENERALIZATION ANALYSIS ({mode.upper()})\n{'='*60}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    summary_csv_path = save_dir / f"{mode}_Performance_Numerical_Results.csv"
    raw_data_csv_path = save_dir / f"{mode}_Raw_Latency_Data.csv"

    violin_df = None
    table_data = None

    # --- A. CHECK CACHE ---
    if summary_csv_path.exists() and raw_data_csv_path.exists():
        print(f"[+] Found cached data at {save_dir}. Loading...")
        violin_df = pd.read_csv(raw_data_csv_path)
        stats_df_loaded = pd.read_csv(summary_csv_path)
        table_data = stats_df_loaded.values.tolist()
    else:
        print(f"[+] No cached data found. Benchmarking {len(all_models)} models...")
        
        performance_results = []
        all_feat_latencies_ms = []
        all_pred_latencies_ms = []

        for i, model in enumerate(all_models):
            print(f"   -> Benchmarking Model {i+1}/{len(all_models)}...")
            result = _run_single_model_benchmark(
                image_df, selected_indices, model, n_augmentations, n_warmup, n_repeats, extraction_mode=mode
            )
            performance_results.append(result)
            
            # Recalculate effective batch size for correct per-image math
            effective_batch_size = len(image_df) + (len(image_df) * n_augmentations) if n_augmentations > 0 else len(image_df)

            batch_feat_times = np.array(result['feat_times_raw'])
            batch_pred_times = np.array(result['pred_times_raw'])
            
            per_img_feat_ms = (batch_feat_times / effective_batch_size) * 1000
            per_img_pred_ms = (batch_pred_times / effective_batch_size) * 1000
            
            all_feat_latencies_ms.extend(per_img_feat_ms)
            all_pred_latencies_ms.extend(per_img_pred_ms)
        
        perf_df = pd.DataFrame(performance_results)

        # Prepare Violin DataFrame
        violin_df = pd.DataFrame({
            'Latency (ms)': np.concatenate([all_feat_latencies_ms, all_pred_latencies_ms]),
            'Component': ['Feature Extraction'] * len(all_feat_latencies_ms) + 
                         ['Prediction'] * len(all_pred_latencies_ms)
        })
        
        # Prepare Table Data
        mean_ips = perf_df['throughput_ips'].mean()
        mean_total_lat_ms = perf_df['total_latency_ms'].mean()
        std_total_lat_ms = perf_df['total_latency_ms'].std()
        
        mean_feat_ms = np.mean(all_feat_latencies_ms)
        std_feat_ms = np.std(all_feat_latencies_ms)
        mean_pred_ms = np.mean(all_pred_latencies_ms)
        std_pred_ms = np.std(all_pred_latencies_ms)
        
        table_data = [
            ['Throughput (Images / Second)', f"{mean_ips:.1f} ± {perf_df['throughput_ips'].std():.1f}"],
            ['Total Per-Image Latency (ms)', f"{mean_total_lat_ms:.2f} ± {std_total_lat_ms:.2f}"],
            ['Per-Image Feature Extraction (ms)', f"{mean_feat_ms:.2f} ± {std_feat_ms:.2f}"],
            ['Per-Image Prediction (ms)', f"{mean_pred_ms:.2f} ± {std_pred_ms:.2f}"],
            ['Effective Runs ($n_{repeats}$)', f"{n_repeats}"]
        ]

        # Save
        print("[+] Saving calculated data...")
        violin_df.to_csv(raw_data_csv_path, index=False)
        stats_df = pd.DataFrame(table_data, columns=['Metric', 'Value'])
        stats_df.to_csv(summary_csv_path, index=False)

    # --- B. VISUALIZATION (Nature Style with Journal Table) ---
    print("\n[+] Generating consolidated performance dashboard...")
    set_nature_style(font_size=14)
    
    fig = plt.figure(figsize=(20, 9))
    # Adjust width ratios to give table appropriate space
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.2)

    # --- Left: Violin Plot ---
    ax_left = fig.add_subplot(gs[0])
    
    sns.violinplot(
        data=violin_df, 
        x='Component', 
        y='Latency (ms)', 
        ax=ax_left, 
        color='skyblue', 
        inner='quartile',
        linewidth=1.2,
        saturation=0.9
    )

    ax_left.set_title('A. Average Per-Image Latency Breakdown', fontsize=20, weight='bold', loc='left', pad=15)
    ax_left.set_ylabel('Time (ms)', fontsize=18)
    ax_left.set_xlabel('') 
    
    # Force black spines/ticks
    ax_left.spines['left'].set_visible(True)
    ax_left.spines['bottom'].set_visible(True)
    ax_left.spines['left'].set_color('black')
    ax_left.spines['bottom'].set_color('black')
    ax_left.tick_params(axis='both', colors='black', labelcolor='black', width=1)

    # --- Right: Journal Style Table ---
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    
    col_labels = ['Performance Metric', 'Value (Mean ± Std Dev)']
    
    # Create the table
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.6, 0.4],
        bbox=[0, 0.2, 1, 0.6] # Adjust bbox to center table vertically in subplot
    )

    # --- APPLYING JOURNAL STYLE TO TABLE ---
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    
    # Title B
    ax_table.text(0.0, 0.85, 'B. Performance Statistics', fontsize=20, weight='bold', transform=ax_table.transAxes)

    # Clean styling (No vertical lines, just bold horizontal separators)
    for (row, col), cell in table.get_celld().items():
        # Clear default borders
        cell.set_edgecolor('white')
        cell.set_linewidth(0)
        
        # Header Styling
        if row == 0:
            cell.set_text_props(weight='bold')
            # Add top/bottom lines to header manually if needed, or rely on drawing lines below
        else:
            cell.set_text_props(color='black')
    
    # DRAW LINES (The Nature Look)
    # Top Line (Above Header)
    ax_table.plot([0, 1], [0.8, 0.8], color='black', lw=2, transform=ax_table.transAxes)
    # Header Separator (Below Header)
    # Approx calculation: bbox height is 0.6. Header is 1 row out of 6 (5 data + 1 header).
    # Header bottom ≈ 0.8 - (0.6 / 6) = 0.7
    row_height = 0.6 / len(table_data) # Approximation
    header_bottom = 0.8 - 0.1 # Approximate visual adjustment
    ax_table.plot([0, 1], [0.7, 0.7], color='black', lw=1, transform=ax_table.transAxes)
    # Bottom Line (Below Data)
    ax_table.plot([0, 1], [0.2, 0.2], color='black', lw=2, transform=ax_table.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    img_save_path = save_dir / f"{mode}_Performance_Dashboard.png"
    plt.savefig(img_save_path, bbox_inches='tight', dpi=300)
    print(f"[+] Graph saved to: {img_save_path}")
    plt.show()

def generate_comparative_scalability_dashboard(
        save_dir,
        fit_ = False,
        ticks_fontsize = 16,
        axis_label_fontsize = 16,
        title_label_fontsize = 20,
        table_text_size = 16
    ):
    print(f"\n{'='*60}\nGENERATING COMPARATIVE SCALABILITY DASHBOARD (VISIBLE TICKS)\n{'='*60}")
    
    # Set Style
    set_nature_style()
    
    save_dir = Path(save_dir)
    modes = ['serial', 'parallel']
    data = {}
    
    # --- 1. LOAD DATA ---
    for mode in modes:
        csv_path = save_dir / f"{mode}_Scalability_Data_Aggregated.csv"
        if not csv_path.exists():
            print(f"[!] Error: Missing data for {mode} mode at {csv_path}")
            return
        
        # Load and group
        df = pd.read_csv(csv_path)
        grouped = df.groupby('Batch Size').agg({
            'Total Time (s)': ['mean', 'std'],
            'Throughput (img/s)': ['mean', 'std']
        }).reset_index()
        
        grouped.columns = ['Batch Size', 'Time_Mean', 'Time_Std', 'IPS_Mean', 'IPS_Std']
        data[mode] = grouped

    # --- 2. CONFIGURATION ---
    styles = {
        'serial':   {'color': '#1f77b4', 'fmt': 'o--', 'label': 'Serial (1 Core)'},
        'parallel': {'color': '#2ca02c', 'fmt': 's-',  'label': 'Parallel (Multi-Core)'}
    }
    
    # ==========================================
    # 3. HELPER FUNCTIONS
    # ==========================================
    def get_power_fit(x, y):
        if len(x) < 2: return None, None, ""
        slope, intercept, r_val, _, _ = stats.linregress(np.log(x), np.log(y))
        a = np.exp(intercept)
        b = slope
        x_plot = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        y_plot = a * (x_plot ** b)
        eq_text = f"$T \\approx {a:.2e} N^{{{b:.2f}}}$"
        return x_plot, y_plot, eq_text

    def format_scientific(x, pos):
        if x == 0: return "0"
        exponent = int(np.floor(np.log10(x)))
        coeff = x / (10**exponent)
        if abs(coeff - 1.0) < 0.01: return f"$10^{{{exponent}}}$"
        return f"${int(coeff)}\\times 10^{{{exponent}}}$"
        
    def format_simple_float(x, pos):
            return f"{x:g}"
            
    # ==========================================
    # 4. PLOTTING GRAPHS
    # ==========================================
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 16))
    plt.subplots_adjust(bottom=0.45, top=0.92, wspace=0.2) 
    
    # --- GRAPH A: TIME SCALING ---
    ax_time = axes[0]
    
    for mode in modes:
        df = data[mode]
        st = styles[mode]
        ax_time.errorbar(
            df['Batch Size'], df['Time_Mean'], yerr=df['Time_Std'],
            fmt=st['fmt'], color=st['color'], label=st['label'],
            capsize=4, markersize=8, linewidth=2, alpha=0.9
        )
        if fit_:
            x_fit, y_fit, eq = get_power_fit(df['Batch Size'], df['Time_Mean'])
            if x_fit is not None:
                ax_time.plot(x_fit, y_fit, color=st['color'], linestyle=':', linewidth=1.5, alpha=0.8)
                ax_time.text(x_fit[-1], y_fit[-1]*0.8, eq, color=st['color'], fontsize=12, ha='right')

    ax_time.set_xscale('log')
    ax_time.set_yscale('log')
    ax_time.set_title('A. Processing Time Scaling', fontsize=title_label_fontsize, weight='bold', loc='left')
    ax_time.set_ylabel('Total Execution Time (s)', fontsize=axis_label_fontsize)
    ax_time.set_xlabel('Batch Size (Number of Images)', fontsize=axis_label_fontsize)
    
    # --- TICKS CONFIGURATION (FORCED) ---
    ax_time.grid(False) 
    
    # 1. Force Minor Ticks on Log Scale
    ax_time.minorticks_on()
    
    # 2. Major Ticks: Make them long (length=8) and thick (width=1.5)
    ax_time.tick_params(
        axis='both', which='major', direction='out', 
        length=8, width=1.5, colors='black',
        left=True, bottom=True, labelsize=ticks_fontsize
    )
    
    # 3. Minor Ticks: Make them visible (length=4)
    ax_time.tick_params(
        axis='both', which='minor', direction='out', 
        length=4, width=1.0, colors='black',
        left=True, bottom=True, labelsize=ticks_fontsize
    )

    ax_time.get_xaxis().set_major_formatter(FuncFormatter(format_scientific))
    ax_time.yaxis.set_major_formatter(FuncFormatter(format_simple_float))
    
    # Black Spines
    for spine in ax_time.spines.values(): 
        spine.set_color('black')
        spine.set_linewidth(1.2)

    # --- GRAPH B: THROUGHPUT ---
    ax_ips = axes[1]
    
    for mode in modes:
        df = data[mode]
        st = styles[mode]
        ax_ips.errorbar(
            df['Batch Size'], df['IPS_Mean'], yerr=df['IPS_Std'],
            fmt=st['fmt'], color=st['color'], label=st['label'],
            capsize=4, markersize=8, linewidth=2, alpha=0.9
        )

    ax_ips.set_xscale('log')
    ax_ips.set_title('B. Throughput Stability', fontsize=title_label_fontsize, weight='bold', loc='left')
    ax_ips.set_ylabel('Images / Sec', fontsize=axis_label_fontsize)
    ax_ips.set_xlabel('Batch Size (Number of Images)', fontsize=axis_label_fontsize)
    
    # --- TICKS CONFIGURATION (FORCED) ---
    ax_ips.grid(False)
    ax_ips.minorticks_on()
    
    # Major Ticks
    ax_ips.tick_params(
        axis='both', which='major', direction='out', 
        length=8, width=1.5, colors='black',
        left=True, bottom=True, labelsize=ticks_fontsize
    )
    
    # Minor Ticks
    ax_ips.tick_params(
        axis='both', which='minor', direction='out', 
        length=4, width=1.0, colors='black',
        left=True, bottom=True, labelsize=ticks_fontsize
    )
    
    ax_ips.get_xaxis().set_major_formatter(FuncFormatter(format_scientific))
    
    # Black Spines
    for spine in ax_ips.spines.values(): 
        spine.set_color('black')
        spine.set_linewidth(1.2)

    # --- 5. LEGEND (Centered) ---
    custom_lines = [
        Line2D([0], [0], color=styles['serial']['color'], marker='o', linestyle='--', markersize=8, lw=2),
        Line2D([0], [0], color=styles['parallel']['color'], marker='s', linestyle='-', markersize=8, lw=2)
    ]
    
    fig.legend(
        custom_lines, 
        [styles['serial']['label'], styles['parallel']['label']], 
        loc='center', 
        bbox_to_anchor=(0.5, 0.38), 
        ncol=2, 
        fontsize=16,
        frameon=False
    )

    # ==========================================
    # 6. UNIFIED TABLE (Nature Style)
    # ==========================================
    fig.text(0.1, 0.32, "C. Numerical Results Comparison", ha='left', fontsize=title_label_fontsize, weight='bold')

    df_s = data['serial'].copy()
    df_p = data['parallel'].copy()
    
    df_s['Time_Str'] = df_s.apply(lambda r: f"{r['Time_Mean']:.2f} ± {r['Time_Std']:.2f}", axis=1)
    df_s['IPS_Str']  = df_s.apply(lambda r: f"{r['IPS_Mean']:.1f} ± {r['IPS_Std']:.1f}", axis=1)
    df_p['Time_Str'] = df_p.apply(lambda r: f"{r['Time_Mean']:.2f} ± {r['Time_Std']:.2f}", axis=1)
    df_p['IPS_Str']  = df_p.apply(lambda r: f"{r['IPS_Mean']:.1f} ± {r['IPS_Std']:.1f}", axis=1)
    
    merged = pd.merge(df_s[['Batch Size', 'Time_Str', 'IPS_Str']], 
                      df_p[['Batch Size', 'Time_Str', 'IPS_Str']], 
                      on='Batch Size', suffixes=('_S', '_P'))
    
    table_data = []
    for _, row in merged.iterrows():
        table_data.append([
            int(row['Batch Size']),
            row['Time_Str_S'],      
            row['Time_Str_P'],      
            row['IPS_Str_S'],       
            row['IPS_Str_P']        
        ])
        
    ax_table = plt.axes([0.1, 0.02, 0.8, 0.28]) 
    ax_table.axis('off')

    col_labels = ['Images (N)', 'Time (Serial)', 'Time (Parallel)', 'IPS (Serial)', 'IPS (Parallel)']
    
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(table_text_size)
    
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('white') 
        cell.set_linewidth(0)
        
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_text_props(color='black')

    # Horizontal Lines
    ax_table.plot([0, 1], [1, 1], color='black', lw=2, transform=ax_table.transAxes, clip_on=False)
    n_rows = len(table_data) + 1
    row_h = 1.0 / n_rows
    line_y = 1.0 - row_h
    ax_table.plot([0, 1], [line_y, line_y], color='black', lw=1, transform=ax_table.transAxes)
    ax_table.plot([0, 1], [0, 0], color='black', lw=2, transform=ax_table.transAxes)

    # Save
    out_path = save_dir / "Comparative_Scalability_Analysis_Ticks.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[+] Comparative Dashboard saved to: {out_path}")
    plt.show()

def generate_consolidated_dashboard(
    save_dir,
    # --- Font Controls ---
    font_title=26,
    font_label=26,
    font_tick=20,
    font_table=27
):
    save_dir = Path(save_dir)
    modes = ['serial', 'parallel']
    data = {}

    print(f"\n{'='*60}\nGENERATING CONSOLIDATED DASHBOARD (WITH GRIDS)\n{'='*60}")
    set_nature_style(font_size=12)

    # --- 1. LOAD DATA ---
    for mode in modes:
        raw_path = save_dir / f"{mode}_Raw_Latency_Data.csv"
        summary_path = save_dir / f"{mode}_Performance_Numerical_Results.csv"
        
        if not raw_path.exists() or not summary_path.exists():
            print(f"[!] Error: Missing data for mode '{mode}' in {save_dir}")
            return

        data[mode] = {
            'raw': pd.read_csv(raw_path),
            'summary': pd.read_csv(summary_path)
        }

    # --- 2. CALCULATE SHARED LIMITS ---
    all_latencies = pd.concat([data['serial']['raw']['Latency (ms)'], 
                               data['parallel']['raw']['Latency (ms)']])
    y_max = all_latencies.max() * 1.15
    
    # --- 3. PLOT SETUP ---
    fig = plt.figure(figsize=(20, 14))
    
    # Grid: Top row (Plots split), Bottom row (Single Table spanning all)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.6], hspace=0.3, wspace=0.15)

    configs = {
        'serial':   {'color': 'skyblue',   'title': 'A. Serial (Single Core)',   'col_idx': 0},
        'parallel': {'color': 'lightgreen', 'title': 'B. Parallel (Multi-Core)', 'col_idx': 1}
    }

    # --- 4. PLOTTING LOOP (Violins) ---
    for mode, cfg in configs.items():
        col = cfg['col_idx']
        ax_plot = fig.add_subplot(gs[0, col])
        
        sns.violinplot(
            data=data[mode]['raw'], x='Component', y='Latency (ms)',
            ax=ax_plot, color=cfg['color'], inner='quartile', linewidth=1.5, alpha=0.9, saturation=0.8
        )
        
        ax_plot.set_title(cfg['title'], fontsize=font_title, weight='bold', pad=20, loc='left')
        
        # --- Axis Customization ---
        ax_plot.set_ylim(bottom=-0.1, top=y_max)
        
        def non_negative_formatter(x, pos):
            return f"{x:g}" if x >= 0 else ""
            
        ax_plot.yaxis.set_major_formatter(FuncFormatter(non_negative_formatter))
        ax_plot.set_xlabel('') 

        # Spines
        ax_plot.spines['left'].set_visible(True)
        ax_plot.spines['bottom'].set_visible(True)
        ax_plot.spines['left'].set_color('black')
        ax_plot.spines['bottom'].set_color('black')
        ax_plot.spines['left'].set_linewidth(1.5)
        ax_plot.spines['bottom'].set_linewidth(1.5)
        ax_plot.set_ylabel('Latency (ms)', fontsize=font_label)
        ax_plot.tick_params(axis='y', labelsize=font_tick, colors='black', left=True, length=6, width=1.2)
        ax_plot.tick_params(axis='x', labelsize=font_tick, colors='black', bottom=True, length=6, width=1.2)
        
        # --- ENABLE GRID (TICKS LINE) ---
        # This adds the dotted lines corresponding to the ticks
        ax_plot.grid(True, axis='both', linestyle=':', linewidth=0.8, color='gray', alpha=0.5)

    # --- 5. UNIFIED TABLE (Nature Style) ---
    ax_table = fig.add_subplot(gs[1, :]) 
    ax_table.axis('off')
    
    # Merge DataFrames
    df_serial = data['serial']['summary'].rename(columns={'Value': 'Serial'})
    df_parallel = data['parallel']['summary'].rename(columns={'Value': 'Parallel'})
    
    df_merged = pd.merge(df_serial, df_parallel, on='Metric')
    
    table_vals = df_merged.values.tolist()
    col_labels = ['Metric', 'Serial Execution', 'Parallel Execution']
    
    # Draw Table
    tbl = ax_table.table(
        cellText=table_vals,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.3, 0.3],
        bbox=[0, 0, 1, 0.8] 
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_table)
    
    # --- Styling Loop ---
    for (row, c_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor('white') 
        cell.set_linewidth(0)
        
        # Header Styling
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_text_props(color='black')
            if c_idx == 0:
                cell.set_text_props(ha='left', weight='bold')

    # --- DRAW HORIZONTAL LINES ---
    # Top Line
    ax_table.plot([0, 1], [0.8, 0.8], color='black', lw=2, transform=ax_table.transAxes)
    
    # Header Separator
    n_rows = len(table_vals) + 1 
    row_h = 0.8 / n_rows
    line_y = 0.8 - row_h
    ax_table.plot([0, 1], [line_y, line_y], color='black', lw=1, transform=ax_table.transAxes)
    
    # Bottom Line
    ax_table.plot([0, 1], [0, 0], color='black', lw=2, transform=ax_table.transAxes)

    # Title C
    ax_table.text(0.0, 0.85, 'C. Performance Benchmark Statistics', 
                 fontsize=font_title, weight='bold', transform=ax_table.transAxes)

    # Output
    output_path = save_dir / "Unified_Performance_Dashboard_Nature.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[+] Unified Dashboard saved to: {output_path}")
    plt.show()


# =======================================================================================================
# New RealTime Engine 
# =======================================================================================================

#_____________________________________________________________________________________
# 1. HELPER FUNCTIONS

def load_all_resources(seeds, results_root):
    """
    Loads Models and Data. 
    Since features are identical, we build one 'X_mega' for diversity checks.
    """
    print(f"\n{'='*60}\nSTEP 1: LOADING RESOURCES\n{'='*60}")
    
    models = {}
    scalers = {}
    selectors = {} 
    test_sets = {} 
    
    X_parts = [] # To build Mega-Set
    
    for seed in seeds:
        base_path = Path(results_root) / f"Results_{seed}"
        path_model = base_path / f"Results/Final_Results/Test_Results/{MODEL_TO_ANALYZE}_final_model.pkl"
        path_scaler = base_path / "Datasets/Training_Test_Splits/Global_Data/Global_Scalers/global_scaler.pkl"
        path_feat = base_path / "Datasets/Training_Test_Splits/features_to_keep.pkl"
        path_test = base_path / "Datasets/Training_Test_Splits/Global_Data/test/test_scaled.pkl"
        
        # Load Model
        if path_model.exists():
            m = joblib.load(path_model)
            if hasattr(m, "best_estimator_"): m = m.best_estimator_
            models[seed] = m
        else:
            continue

        # Load Scaler
        if path_scaler.exists():
            scalers[seed] = joblib.load(path_scaler)
            
        # Load Data (Only if we haven't loaded this specific file before?)
        # Actually, if seeds imply different data splits, we load all.
        if path_feat.exists() and path_test.exists():
            feat = joblib.load(path_feat)
            df_test = joblib.load(path_test)
            
            # Store per-seed data for Robustness (Self-Test)
            selectors[seed] = feat
            test_sets[seed] = (df_test[feat], df_test['label'])
            
            # Add to Mega-Set parts
            X_parts.append(df_test[feat])
            
    # Create the Common Reference Set (Mega-Set)
    if X_parts:
        X_mega = pd.concat(X_parts, ignore_index=True)
    else:
        X_mega = pd.DataFrame()
            
    print(f"[+] Successfully loaded {len(models)} Candidates.")
    print(f"[+] Mega-Set Size: {X_mega.shape}")
    return models, scalers, selectors, test_sets, X_mega

def run_robustness_check(models, test_sets):
    """
    Evaluates each model on its own held-out test set (Self-Consistency).
    """
    print(f"\n{'='*60}\nSTEP 2: ROBUSTNESS RANKING\n{'='*60}")
    
    scores = {}

    for seed, model in models.items():
        if seed in test_sets:
            X_t, y_t = test_sets[seed]
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_t)[:, 1]
                else:
                    probs = model.predict(X_t)
                
                auc = roc_auc_score(y_t, probs)
                scores[seed] = auc
                print(f"   Model {seed} AUC: {auc:.5f}")
            except Exception as e:
                print(f"   [!] Model {seed} failed: {e}")
                scores[seed] = 0.5
    
    robustness_series = pd.Series(scores)
    print("\n🏆 TOP 5 PERFORMANCE:")
    print(robustness_series.sort_values(ascending=False).head(5))
    return robustness_series

def calculate_diversity_matrix(models, X_mega, plot=False):
    """
    Calculates correlation on the Mega-Set.
    Since features are identical, this is fast and simple.
    """
    print(f"\n{'='*60}\nSTEP 3: DIVERSITY CHECK\n{'='*60}")
    
    all_preds = pd.DataFrame()
    print(f"   Generating predictions for {len(models)} models on {len(X_mega)} samples...")
    
    for seed, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_mega)[:, 1]
            else:
                preds = model.predict(X_mega)
            all_preds[seed] = preds
        except Exception as e:
            print(f"   [!] Skipping Seed {seed}: {e}")

    corr_matrix = all_preds.corr()
    
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=0.9, vmax=1.0)
        plt.title("Model Correlation Matrix")
        plt.show()
        
    return corr_matrix

def select_golden_trio(robustness_scores, corr_matrix, models, printf_=False):
    print(f"\n{'='*60}\nSTEP 4: SELECTION ALGORITHM\n{'='*60}")
    
    selected = []
    scaler = MinMaxScaler()
    candidate_pool = list(models.keys())

    def get_scores(candidates, current_selection):
        raw_auc = []
        raw_div = []
        
        for c in candidates:
            # AUC
            raw_auc.append(robustness_scores.get(c, 0))
            
            # Diversity
            if not current_selection:
                raw_div.append(1.0)
            else:
                # Calculate average correlation to CURRENTLY selected models
                # We want MINIMAL correlation (1 - corr)
                corrs = []
                for s in current_selection:
                    if s in corr_matrix.index and c in corr_matrix.columns:
                        corrs.append(corr_matrix.loc[s, c])
                
                if corrs:
                    avg_corr = sum(corrs) / len(corrs)
                    raw_div.append(1.0 - avg_corr) 
                else:
                    raw_div.append(0.0) # Should not happen if matrix is full
        
        # Normalize
        # Handle single candidate or zero variance
        if len(raw_auc) > 1 and np.max(raw_auc) > np.min(raw_auc):
            norm_auc = scaler.fit_transform(np.array(raw_auc).reshape(-1, 1)).flatten()
        else:
            norm_auc = np.array(raw_auc)

        if current_selection and len(raw_div) > 1 and np.max(raw_div) > np.min(raw_div):
            norm_div = scaler.fit_transform(np.array(raw_div).reshape(-1, 1)).flatten()
        else:
            norm_div = np.zeros(len(candidates))

        scores = {}
        for i, c in enumerate(candidates):
            if not current_selection:
                scores[c] = raw_auc[i]
            else:
                # 50% Performance / 50% Diversity
                scores[c] = (norm_auc[i] * 0.5) + (norm_div[i] * 0.5)
        return scores

    # 1. Anchor (Best AUC)
    best = robustness_scores.idxmax()
    selected.append(best)
    if printf_:
        print(f"1. 🥇 Anchor Model:    Seed {best} (AUC: {robustness_scores[best]:.5f})")

    # 2. Partner
    remaining = [x for x in candidate_pool if x not in selected]
    if remaining:
        s2 = get_scores(remaining, selected)
        best_2 = max(s2, key=s2.get)
        selected.append(best_2)
        if printf_:
            print(f"2. 🥈 Divergent Model: Seed {best_2} (Score: {s2[best_2]:.4f})")

    # 3. Third
    remaining = [x for x in candidate_pool if x not in selected]
    if remaining:
        s3 = get_scores(remaining, selected)
        best_3 = max(s3, key=s3.get)
        selected.append(best_3)
        if printf_:
            print(f"3. 🥉 Divergent Model: Seed {best_3} (Score: {s3[best_3]:.4f})")

    return selected

def setup_engine_workspace(models_dict, scalers_dict, selectors_dict, selected_seeds):
    print(f"\n{'='*60}\nSTEP 5: ENGINE WORKSPACE DEPLOYMENT\n{'='*60}")
    
    BASE_DIR = Path("Engines")
    RT_DIR = BASE_DIR / "Original_RealTimeFrameClassification"
    
    # 1. Cleanup
    for target_dir in [RT_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        old_assets = list(target_dir.glob("rt_asset_seed_*.pkl"))
        if old_assets:
            for f in old_assets:
                try: f.unlink()
                except: pass

    # 2. Deploy
    print(f"[+] Deploying Seeds: {selected_seeds}")
    saved_count = 0
    
    for seed in selected_seeds:
        model = models_dict[seed]
        scaler = scalers_dict.get(seed)
        selector = selectors_dict.get(seed) # Guaranteed to exist
        
        # A. Extract Booster
        if hasattr(model, 'booster_'): booster = model.booster_
        else: booster = model
            
        # B. Force Single Threading
        if hasattr(booster, 'params'):
            booster.params['n_jobs'] = 1
            booster.params['num_threads'] = 1
            
        # C. Prepare Scaler
        if hasattr(scaler, 'mean_'):
            mean_32 = scaler.mean_.astype(np.float32)
            scale_32 = scaler.scale_.astype(np.float32)
        else:
            mean_32 = np.array([0.0], dtype=np.float32)
            scale_32 = np.array([1.0], dtype=np.float32)
            
        # D. Save
        asset_name = f"rt_asset_seed_{seed}.pkl"
        payload = {
            'booster': booster,
            'scaler_mean': mean_32,
            'scaler_scale': scale_32,
            'features': selector
        }
        
        joblib.dump(payload, RT_DIR / asset_name)
        print(f"   -> Deployed Asset {seed}: {asset_name}")
        saved_count += 1
        
    print(f"\n[+] Deployment Complete. {saved_count} assets ready.")

def measure_peak_ram_mb(func, frames):
    """Returns (latencies, outputs, peak_ram_mb) using precise Python tracemalloc."""
    gc.collect()
    tracemalloc.start()
    
    latencies, outputs = [], []
    for img in frames:
        t0 = perf_counter()
        res = func(img)
        t1 = perf_counter()
        if res is not None:
            latencies.append((t1 - t0) * 1000)
            outputs.append(res)
            
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return latencies, outputs, peak_bytes / (1024 * 1024)  # -> MB

#_____________________________________________________________________________________
# 2. DASHBOARD GENERATION

def generate_scientific_dashboard(
    df_results, 
    df_metrics, 
    save_path=None, 
    n_stability_frames=200,
    save_plot=True,
    title_fs=20,
    label_fs=16,
    tick_fs=14
):
    print(f"\n{'='*60}\nGENERATING SCIENTIFIC DASHBOARD (NATURE STYLE)\n{'='*60}")
    
    sns.set_context("talk")
    sns.set_style("white")
    
    fig = plt.figure(figsize=(26, 16)) 
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.6], hspace=0.35, wspace=0.2)

    # --- PANEL A: Latency Distribution ---
    ax_dist = fig.add_subplot(gs[0, 0])
    for label, color in COLORS.items():
        subset = df_results[df_results['Engine'] == label]
        if not subset.empty:
            sns.kdeplot(data=subset, x='Latency (ms)', fill=True, alpha=0.3, 
                        color=color, label=label, linewidth=2.5, ax=ax_dist, log_scale=True)
    
    ax_dist.set_title("A. Latency Distribution (Log Scale)", fontsize=title_fs, weight='bold', loc='left')
    ax_dist.set_xlabel("Latency (ms)", fontsize=label_fs)
    ax_dist.set_ylabel("Density", fontsize=label_fs)
    
    ax_dist.grid(False)
    ax_dist.minorticks_on()
    ax_dist.tick_params(axis='both', which='major', length=8, width=1.2, labelsize=tick_fs, colors='black', left=True, bottom=True)
    for spine in ax_dist.spines.values(): spine.set_color('black')
    ax_dist.legend(fontsize=tick_fs, frameon=False)

    # --- PANEL B: Stability (Controlled Zoom) ---
    ax_jitter = fig.add_subplot(gs[0, 1])
    df_zoom = df_results[df_results['Frame'] < n_stability_frames]
    
    sns.lineplot(data=df_zoom, x='Frame', y='Latency (ms)', hue='Engine', palette=COLORS, 
                 linewidth=2, ax=ax_jitter, errorbar='sd')
    
    ax_jitter.set_title(f"B. Processing Stability (First {n_stability_frames} Frames)", fontsize=title_fs, weight='bold', loc='left')
    ax_jitter.set_ylabel("Latency (ms)", fontsize=label_fs)
    ax_jitter.set_xlabel("Frame Index", fontsize=label_fs)
    
    ax_jitter.grid(False)
    ax_jitter.minorticks_on()
    ax_jitter.tick_params(axis='both', which='major', length=8, width=1.2, labelsize=tick_fs, colors='black', left=True, bottom=True)
    for spine in ax_jitter.spines.values(): spine.set_color('black')
    ax_jitter.legend(loc='upper right', fontsize=tick_fs, frameon=False)

    # --- PANEL C: Statistical Table ---
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis('off')
    
    col_labels = ['Pipeline Version', 'Median Latency', 'Throughput', 'Peak RAM (Python)', 'Fidelity (Corr)', 'Avg Error (MAE)', 'Label Agreement']
    table_cells = []
    
    for label in df_metrics['Engine'].unique():
        row_data = df_metrics[df_metrics['Engine'] == label].iloc[0]
        
        def fmt(val, err, unit=""): 
            return f"{val:.2f} ± {err:.2f}{unit}"
        
        agreement_pct = row_data['Agreement_Mean'] * 100
        
        if "Old" in label:
            fid_str, mae_str, agr_str = "1.00", "-", "-"
        else:
            fid_str = f"{row_data['Fidelity_Mean']:.2f}"
            mae_str = f"{row_data['MAE_Mean']:.4f}"
            agr_str = f"{agreement_pct:.2f}%"

        cells = [
            label,
            fmt(row_data['Median_Mean'], row_data['Median_Std'], " ms"),
            fmt(row_data['FPS_Mean'], row_data['FPS_Std'], " FPS"),
            fmt(row_data['RAM_Mean'], row_data['RAM_Std'], " MB"), 
            fid_str,
            mae_str,
            agr_str
        ]
        table_cells.append(cells)

    table = ax_stats.table(cellText=table_cells, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0, 0, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(label_fs)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(0)
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_text_props(color='black')
            if col == 0: cell.set_text_props(weight='bold')

    ax_stats.plot([0, 1], [0.8, 0.8], color='black', lw=2, transform=ax_stats.transAxes)
    line_y = 0.8 - (0.8 / (len(table_cells) + 1))
    ax_stats.plot([0, 1], [line_y, line_y], color='black', lw=1, transform=ax_stats.transAxes)
    ax_stats.plot([0, 1], [0, 0], color='black', lw=2, transform=ax_stats.transAxes)

    ax_stats.text(0.0, 0.85, "C. Performance, Resource Usage, & Equivalence Statistics", 
                  fontsize=title_fs, weight='bold', transform=ax_stats.transAxes)
    
    if save_plot and save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=IMAGE_DPI, bbox_inches='tight')
        print(f"[+] Dashboard saved to {save_path}")
        
    plt.show()

#_____________________________________________________________________________________
# 3. BENCHMARK EXECUTION LOGIC

def format_summary_table(df):
    """Formats the numeric DataFrame into a beautiful string table with ± for the console."""
    display_df = pd.DataFrame()
    display_df['Engine'] = df['Engine']
    
    def fmt(row, col_base, decimals=2):
        val = row.get(col_base)
        if pd.isna(val): return "NaN"
        if isinstance(val, str): return val
            
        std_col = f'{col_base} Std'
        if std_col in df.columns:
            std_val = row.get(std_col)
            if not pd.isna(std_val) and not isinstance(std_val, str):
                return f"{float(val):.{decimals}f} ± {float(std_val):.{decimals}f}"
                
        return f"{float(val):.{decimals}f}"

    display_df['Median Latency (ms)'] = df.apply(lambda r: fmt(r, 'Median_Mean', 2), axis=1) if 'Median_Mean' in df else df.apply(lambda r: fmt(r, 'Median', 2), axis=1)
    display_df['FPS'] = df.apply(lambda r: fmt(r, 'FPS_Mean', 1), axis=1) if 'FPS_Mean' in df else df.apply(lambda r: fmt(r, 'FPS', 1), axis=1)
    
    if 'Fidelity_Mean' in df.columns or 'Fidelity' in df.columns:
        fid_col = 'Fidelity_Mean' if 'Fidelity_Mean' in df.columns else 'Fidelity'
        mae_col = 'MAE_Mean' if 'MAE_Mean' in df.columns else 'MAE'
        agr_col = 'Agreement_Mean' if 'Agreement_Mean' in df.columns else 'Agreement'
        
        display_df['Fidelity'] = df[fid_col].apply(lambda x: x if isinstance(x, str) else (f"{float(x):.4f}" if not pd.isna(x) else "NaN"))
        display_df['MAE'] = df[mae_col].apply(lambda x: x if isinstance(x, str) else (f"{float(x):.4f}" if not pd.isna(x) else "NaN"))
        display_df['Agreement'] = df[agr_col].apply(lambda x: x if isinstance(x, str) else (f"{float(x):.2f}%" if not pd.isna(x) else "NaN"))
        
    ram_col = 'RAM_Mean' if 'RAM_Mean' in df.columns else 'RAM_Usage'
    display_df['Peak RAM (MB)'] = df.apply(lambda r: fmt(r, ram_col, 1), axis=1)
    
    return display_df

def run_scientific_benchmark(
    n_repetitions=5, 
    n_frames=500, 
    n_stability_frames=200, 
    output_dir="Dataset++/Articles_Images/Speed_Testing_Real_Time_Pipeline",
    force_rerun=False,
    save_results=True,
    title_fs=22,
    label_fs=16,
    tick_fs=14
):
    print(f"\n{'='*60}\nSTARTING ADVANCED SCIENTIFIC BENCHMARK\n{'='*60}")
    
    save_dir = Path(output_dir)
    raw_csv_path = save_dir / "benchmark_raw_data_final.csv"
    metrics_csv_path = save_dir / "benchmark_metrics_final.csv"
    plot_path = save_dir / "Architecture_Speed_Comparison.png"

    # --- CACHE CHECK ---
    if save_results and not force_rerun and raw_csv_path.exists() and metrics_csv_path.exists():
        print(f"[+] Found cached data at {save_dir}. Skipping execution.")
        df_results = pd.read_csv(raw_csv_path)
        df_metrics_agg = pd.read_csv(metrics_csv_path)
        generate_scientific_dashboard(df_results, df_metrics_agg, plot_path, n_stability_frames, save_results, title_fs, label_fs, tick_fs)
        return df_results, df_metrics_agg

    # --- SETUP ENVIRONMENT ---
    workspace_root = Path.cwd()
    if not (workspace_root / "src").exists():
        workspace_root = workspace_root.parent
    rt_engine_dir = (workspace_root / "Engines" / "Original_RealTimeFrameClassification").resolve()
    print(f"[+] Looking for assets in: {rt_engine_dir}")

    if str(rt_engine_dir) not in sys.path: sys.path.append(str(rt_engine_dir))

    try:
        from src.feature_extraction_functions import extract_feats
        from src.Original_RealTimeFrameClassification import load_engine
        from src.dataset_augmentation_functions import create_random_augmented_dataset
    except ImportError as e:
        print(f" [!] Warning: Modules not found. Ensure environment is correct. Error: {e}")
        return None, None

    # --- LOAD RESOURCES ---
    print("\n[+] Loading Resources...")
    RESULTS_ROOT = Path(r'Dataset++')
    MODEL_TO_ANALYZE = 'LGBM'
    SEEDS = range(10)
    
    asset_files = list(rt_engine_dir.glob("rt_asset_seed_*.pkl"))
    active_seeds = [int(re.search(r"seed_(\d+)", p.name).group(1)) for p in asset_files if re.search(r"seed_(\d+)", p.name)]
    
    if not active_seeds: 
        print(" [!] No seeds found in engine folder. Cannot proceed.")
        return None, None

    active_models, scaler_ref = [], None
    for seed in active_seeds:
        p_model = RESULTS_ROOT / f"Results_{seed}/Results/Final_Results/Test_Results/{MODEL_TO_ANALYZE}_final_model.pkl"
        if p_model.is_file():
            m = joblib.load(p_model)
            active_models.append(m.best_estimator_ if hasattr(m, "best_estimator_") else m)
        if scaler_ref is None:
            p_scaler = RESULTS_ROOT / f"Results_{seed}/Datasets/Training_Test_Splits/Global_Data/Global_Scalers/global_scaler.pkl"
            if p_scaler.is_file(): scaler_ref = joblib.load(p_scaler)

    features_to_keep = joblib.load(RESULTS_ROOT / f"Results_{SEEDS[0]}/Datasets/Training_Test_Splits/features_to_keep.pkl")
    new_engine = load_engine(str(rt_engine_dir))

    # --- PREPARE DATA ---
    image_df = joblib.load(RESULTS_ROOT / 'image_df.pkl')
    if len(image_df) < n_frames:
        n_aug = int(np.ceil((n_frames - len(image_df)) / len(image_df)))
        aug_df = create_random_augmented_dataset(image_df, n_augmentations=n_aug, balance=False)
        full_df = pd.concat([image_df, aug_df], ignore_index=True)
        source_images = full_df['image'].sample(n=n_frames, replace=False, random_state=42).tolist()
    else:
        source_images = image_df['image'].sample(n=n_frames, replace=False, random_state=42).tolist()

    print("    -> Pre-calculating baseline probabilities...")
    baseline_probs = []
    for img in source_images:
        f_df = extract_feats(pd.DataFrame({'image': [img], 'label': [0]}), n_jobs=1, verbosity=0)
        seed_probs = [
            active_models[idx].predict_proba(scaler_ref.transform(f_df[features_to_keep]))[:, 1][0] 
            if hasattr(active_models[idx], "predict_proba") 
            else float(active_models[idx].predict(scaler_ref.transform(f_df[features_to_keep]))[0]) 
            for idx in range(len(active_models))
        ]
        baseline_probs.append(np.mean(seed_probs))
        
    baseline_probs = np.array(baseline_probs)
    baseline_labels = (baseline_probs > 0.5).astype(int)
    
    # --- PIPELINES ---
    def old_pipeline(img):
        try:
            f_df = extract_feats(pd.DataFrame({'image': [img], 'label': [0]}), n_jobs=1, verbosity=0)
            vec_scaled = scaler_ref.transform(f_df[features_to_keep]) 
            probs = [m.predict_proba(vec_scaled)[:, 1][0] if hasattr(m, "predict_proba") else float(m.predict(vec_scaled)[0]) for m in active_models]
            return np.mean(probs)
        except: return None

    def new_pipeline(img):
        try: return new_engine.predict_frame(img)
        except: return None

    # --- BENCHMARK LOOP ---
    print(f"\n[+] Running {n_repetitions} Repetitions...")
    results_store, metrics_store = [], []

    for run_idx in range(1, n_repetitions + 1):
        print(f"   -> Run {run_idx}/{n_repetitions}...")
        
        # Track Old Pipeline peak RAM
        gc.collect()
        l_old, o_old, ram_old = measure_peak_ram_mb(old_pipeline, source_images)
        
        # Track New Pipeline peak RAM
        gc.collect()
        l_new, o_new, ram_new = measure_peak_ram_mb(new_pipeline, source_images)
        
        # Equivalence calculations
        min_len = min(len(o_old), len(o_new))
        p_old, p_new = np.array(o_old[:min_len]), np.array(o_new[:min_len])
        b_probs = baseline_probs[:min_len]
        
        if min_len > 10:
            fidelity = np.corrcoef(b_probs, p_new)[0, 1]
            mae = np.mean(np.abs(b_probs - p_new))
            agreement = np.mean((b_probs > 0.5) == (p_new > 0.5))
        else:
            fidelity, mae, agreement = 0.0, 999.0, 0.0

        # Store Latencies
        for i, lat in enumerate(l_old): results_store.append({'Latency (ms)': lat, 'Engine': 'Old Pipeline (Serial)', 'Frame': i, 'Run': run_idx})
        for i, lat in enumerate(l_new): results_store.append({'Latency (ms)': lat, 'Engine': 'New Engine (RealTime)', 'Frame': i, 'Run': run_idx})
        
        # Store Run Metrics using the highly accurate tracemalloc peak
        metrics_store.append({'Engine': 'Old Pipeline (Serial)', 'Median': np.median(l_old), 'FPS': 1000/np.median(l_old), 'RAM_Usage': ram_old, 'Fidelity': 1.0, 'MAE': 0.0, 'Agreement': 1.0})
        metrics_store.append({'Engine': 'New Engine (RealTime)', 'Median': np.median(l_new), 'FPS': 1000/np.median(l_new), 'RAM_Usage': ram_new, 'Fidelity': fidelity, 'MAE': mae, 'Agreement': agreement})

    # --- AGGREGATE ---
    df_results = pd.DataFrame(results_store)
    df_met_raw = pd.DataFrame(metrics_store)

    df_metrics_agg = df_met_raw.groupby('Engine').agg({
        'Median': ['mean', 'std'], 'FPS': ['mean', 'std'], 'RAM_Usage': ['mean', 'std'],
        'Fidelity': ['mean'], 'MAE': ['mean'], 'Agreement': ['mean']
    }).reset_index()
    
    df_metrics_agg.columns = ['Engine', 'Median_Mean', 'Median_Std', 'FPS_Mean', 'FPS_Std', 'RAM_Mean', 'RAM_Std', 'Fidelity_Mean', 'MAE_Mean', 'Agreement_Mean']

    # --- SAVE (Conditioned on save_results) ---
    if save_results:
        print(f"\n[+] Saving Results to {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(raw_csv_path, index=False)
        df_metrics_agg.to_csv(metrics_csv_path, index=False)
    else:
        print("\n[+] Skipping saving raw outputs based on save_results=False flag.")
    
    generate_scientific_dashboard(
        df_results, df_metrics_agg, save_path=plot_path,
        n_stability_frames=n_stability_frames, save_plot=save_results,
        title_fs=title_fs, label_fs=label_fs, tick_fs=tick_fs
    )
    
    return df_results, df_metrics_agg
