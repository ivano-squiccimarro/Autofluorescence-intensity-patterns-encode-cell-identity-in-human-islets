import numpy    as np
import pandas   as pd
import seaborn  as sns

from pathlib    import Path

import matplotlib.pyplot    as plt
import matplotlib.lines     as mlines
import matplotlib.gridspec  as gridspec
from   matplotlib.patches   import Patch
from   matplotlib.lines     import Line2D

from scipy.stats import mannwhitneyu, t

from config.presentation_config import (
    set_nature_style, MODEL_COLORS, CLASS_COLORS, 
    NAME_MAPPING_SHORT, ANALYSIS_METRICS, IMAGE_DPI
)



# ======================================================================================
# Plotting Function for General Results Visualization 
# ======================================================================================

def combined_model_performance_analysis(
    df: pd.DataFrame, 
    metrics: list = None, 
    ref_values: dict = None, 
    stress_level: int = 0,    
    selected_models: list = None,
    # --- LAYOUT CONTROLS ---
    fig_height_multiplier: float = 1.5,
    
    # 1. GRAPH SPACING
    hspace: float = 0.5,                 
    wspace: float = 0.3,                 
    
    # 2. SUPERTITLE 
    title_y: float = 0.96,               
    top_margin: float = 0.90,            
    
    # 3. SECTION SPACING
    section_gap_height: float = 1.5,     
    middle_header_height: float = 0.8,
    title_pad: int = 20,

    # 4. LEGEND SPACING (NEW CONTROLS)
    legend_gap_from_graph: float = 0.8,  
    legend_intra_gap: float = 0.3,       
    # Font Sizes / Widths
    title_fontsize: int = 20,
    header_fontsize: int = 26,             
    tick_fontsize: int = 14, 
    legend_fontsize: int = 14,    
    boxes_linewidth: float = 1.5,
    references_linewiths: float = 1.5,
    legend_line_width: float = 5,
    ball_size: float  = 10,
    
    # Logic Parameters
    max_per_class_diff: float = 0.2,
    # Output
    show: bool = False,
    save: bool = True,
    output_dir: Path = None
):
    
    
    set_nature_style() 

    """
    Generates Nature-style Figures with Model Legend acting as a Divider.
    """
    if metrics is None: metrics = [] 
    if ref_values is None: ref_values = {}
    if output_dir is None: output_dir = Path(".")
    
    # Header Name
    if stress_level is None: header_name = "AVERAGE (TEST + STRESS)"
    elif stress_level == 0:  header_name = "TEST SET"
    else:                    header_name = f"STRESS TEST LEVEL {stress_level}"

    # ==========================================
    # 1. DATA PREPARATION
    # ==========================================
    df = df.copy()
    df.columns = [c.lower() if c in ['AUC', 'PR_AUC'] else c for c in df.columns]
    
    if stress_level is not None:
        analysis_df = df[df['stress_level'] == stress_level].copy()
    else:
        analysis_df = df[df['stress_level'] >= 0].copy()

    if analysis_df.empty:
        print(f"⚠️ Warning: No data found for stress_level={stress_level}")
        return []

    # Filter Models
    if selected_models:
        analysis_df = analysis_df[analysis_df['model_name'].isin(selected_models)]

    # Filter Unbalanced
    if 'precision_1' in analysis_df.columns:
        analysis_df['p_diff'] = (analysis_df['precision_1'] - analysis_df['precision_0']).abs()
        analysis_df['r_diff'] = (analysis_df['recall_1'] - analysis_df['recall_0']).abs()
        stats = analysis_df.groupby('model_name')[['p_diff', 'r_diff']].mean()
        bad_models = stats[(stats['p_diff'] > max_per_class_diff) | (stats['r_diff'] > max_per_class_diff)].index.tolist()
        if len(bad_models) > 0:
            analysis_df = analysis_df[~analysis_df['model_name'].isin(bad_models)]

    if analysis_df.empty: return []

    # SORTING LOGIC
    sort_metric = metrics[0].lower() if metrics else None
    order = sorted(analysis_df['model_name'].unique()) 
    
    if sort_metric and sort_metric in analysis_df.columns:
        if pd.api.types.is_numeric_dtype(analysis_df[sort_metric]):
            order = analysis_df.groupby('model_name')[sort_metric].mean().sort_values(ascending=False).index.tolist()

    # Palette & Mapping Setup (Fixed Scope Issue)
    unique_models = analysis_df['model_name'].unique()
    plot_palette = {}
    
    # Init locals
    final_mapping = {}
    final_class_colors = {}

    try:
        # Try global definitions
        for m in unique_models:
            plot_palette[m] = MODEL_COLORS.get(m, '#333333')
        final_mapping = NAME_MAPPING_SHORT
        final_class_colors = CLASS_COLORS
    except NameError:
        # Fallback
        plot_palette = {m: '#333333' for m in unique_models}
        final_mapping = {m: m for m in unique_models}
        final_class_colors = {'Alpha': 'blue', 'Beta': 'orange'}

    print(f"--- Graphing: {header_name} ---")

    # =========================================================
    # 2. PLOTTING SETUP
    # =========================================================
    n_cols = 3
    n_metrics = len(metrics)
    n_rows_gen = (n_metrics + n_cols - 1) // n_cols if n_metrics > 0 else 1
    
    row_height = 5.0
    
    # --- MACRO LAYOUT (Outer Grid) ---
    top_block_h = n_rows_gen * row_height
    
    # Height Definitions
    leg1_height = 0.5  # Class/Ref Legend (Bottom)
    leg2_height = 1.0  # Model Legend (Middle Divider)
    
    # Bottom Block: Graph + Gap + Legend 1 (Class/Ref)
    # Note: Legend 2 is removed from here
    bot_block_h = row_height + legend_gap_from_graph + leg1_height
    
    # NEW RATIOS: 
    # [Top Graphs, Gap, Model Legend, Gap, Header, Gap, Bottom Graphs]
    outer_ratios = [
        top_block_h, 
        section_gap_height * 0.5, 
        leg2_height,            
        section_gap_height * 0.5, 
        middle_header_height, 
        section_gap_height,   
        bot_block_h
    ]
    
    total_height = sum(outer_ratios) * fig_height_multiplier
    
    fig = plt.figure(figsize=(22, total_height))
    
    fig.suptitle(f"{header_name} PERFORMANCE", fontsize=header_fontsize, 
                 weight='bold', y=title_y)

    # Increased rows to 7 to accommodate the inserted legend
    gs_outer = fig.add_gridspec(7, 1, height_ratios=outer_ratios, 
                                hspace=0.0, 
                                top=top_margin, bottom=0.05, left=0.05, right=0.95)
    
    box_props       = dict(facecolor='none', edgecolor='black', linewidth=boxes_linewidth)
    whisker_props   = dict(linewidth=boxes_linewidth, color='black')
    median_props    = dict(linewidth=boxes_linewidth, color='black')

    # -----------------------------------------------------
    # BLOCK 1: TOP GRAPHS (General Metrics) -> gs_outer[0]
    # -----------------------------------------------------
    plot_idx = 0
    if metrics:
        gs_top = gridspec.GridSpecFromSubplotSpec(n_rows_gen, n_cols, 
                                                  subplot_spec=gs_outer[0], 
                                                  hspace=hspace, wspace=wspace)
        
        for i, metric in enumerate(metrics):
            metric = metric.lower()
            if metric not in analysis_df.columns: continue

            r = i // n_cols
            c = i % n_cols
            ax = fig.add_subplot(gs_top[r, c])

            ax.text(0.0, 1.15, f"{chr(65 + plot_idx)}.", transform=ax.transAxes, 
                    fontsize=title_fontsize, weight='bold')
            plot_idx += 1
            
            sns.stripplot(data=analysis_df, x='model_name', y=metric, order=order,
                          hue='model_name', palette=plot_palette,
                          size=ball_size, alpha=1.0, jitter=0.25, ax=ax, legend=False, zorder=0)
            
            sns.boxplot(data=analysis_df, x='model_name', y=metric, order=order,
                        width=0.5, showfliers=False, ax=ax, zorder=10,
                        boxprops=box_props, whiskerprops=whisker_props, medianprops=median_props)
            
            # Ref Lines
            if metric in ref_values:
                if isinstance(ref_values[metric], dict):
                    ref = (ref_values[metric].get('alpha', 0) + ref_values[metric].get('beta', 0)) / 2
                else:
                    ref = ref_values[metric]
                
                if ref > 0: 
                    ax.axhline(ref, color='red', linestyle='--', linewidth=references_linewiths, alpha=0.7)

            ax.set_title(metric.upper().replace('_', ' '), fontsize=title_fontsize, weight='bold', pad=title_pad)
            ax.set_xlabel(""); ax.set_ylabel("")
            
            short_labels = [final_mapping.get(m, m) for m in order]
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=tick_fontsize)
            
            # --- EVIDENT TICKS ---
            ax.tick_params(axis='both', which='major', width=2, length=6, 
                           labelsize=tick_fontsize, bottom=True, left=True)
            
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # -----------------------------------------------------
    # BLOCK 2: MODEL LEGEND (MIDDLE DIVIDER) -> gs_outer[2]
    # -----------------------------------------------------
    ax_leg_models = fig.add_subplot(gs_outer[2])
    ax_leg_models.axis('off')
    
    model_handles = []
    for model_long in order:
        short = final_mapping.get(model_long, model_long)
        color = plot_palette.get(model_long, '#333333')
        
        # Create a Ball (Marker='o') with the Model's Color
        label_str = f"{short} : {model_long}"
        
        h = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                          markersize=16, label=label_str)
        model_handles.append(h)
    
    ax_leg_models.legend(handles=model_handles, loc='center', ncol=5, 
                         frameon=False, fontsize=legend_fontsize)

    # -----------------------------------------------------
    # BLOCK 3: MIDDLE HEADER -> gs_outer[4]
    # -----------------------------------------------------
    ax_sp = fig.add_subplot(gs_outer[4])
    ax_sp.axis('off')
    ax_sp.text(0.5, 0.5, "PER-CLASS PERFORMANCE BREAKDOWN", ha='center', va='center', 
               fontsize=header_fontsize, weight='bold')

    # -----------------------------------------------------
    # BLOCK 4: BOTTOM GRAPHS + CLASS LEGEND -> gs_outer[6]
    # -----------------------------------------------------
    
    # We only need 3 rows now: Graphs, Gap, Class Legend
    bot_ratios = [row_height, legend_gap_from_graph, leg1_height]
    
    gs_bot = gridspec.GridSpecFromSubplotSpec(3, n_cols, 
                                              subplot_spec=gs_outer[6], 
                                              height_ratios=bot_ratios,
                                              hspace=0.0, 
                                              wspace=wspace)

    pc_metrics = ['precision', 'recall', 'f1']
    pc_palette = {'Alpha': final_class_colors.get('Alpha', 'blue'), 
                  'Beta': final_class_colors.get('Beta', 'orange')}

    # --- ROW 0: GRAPHS ---
    for i, m_base in enumerate(pc_metrics):
        if i >= n_cols: break
        ax = fig.add_subplot(gs_bot[0, i])
        
        ax.text(0.0, 1.10, f"{chr(65 + plot_idx)}.", transform=ax.transAxes, 
                fontsize=title_fontsize, weight='bold')
        plot_idx += 1
        
        col0, col1 = f"{m_base}_0", f"{m_base}_1"
        if col0 in analysis_df.columns:
            melted = analysis_df.melt(id_vars=['model_name'], value_vars=[col0, col1], 
                                      var_name='Class_Type', value_name='Score')
            melted['Class'] = melted['Class_Type'].map({col0: 'Alpha', col1: 'Beta'})
            
            sns.boxplot(data=melted, x='model_name', y='Score', hue='Class', order=order,
                        palette=pc_palette, ax=ax, linewidth=boxes_linewidth, showfliers=False)
        
        if m_base in ref_values:
            if isinstance(ref_values[m_base], dict):
                if 'alpha' in ref_values[m_base]:
                    ax.axhline(ref_values[m_base]['alpha'], color=final_class_colors['Alpha'], ls=':', lw=references_linewiths)
                if 'beta' in ref_values[m_base]:
                    ax.axhline(ref_values[m_base]['beta'], color=final_class_colors['Beta'], ls='--', lw=references_linewiths)

        ax.set_title(f"Per-Class {m_base.capitalize()}", fontsize=title_fontsize, weight='bold', pad=title_pad)
        ax.set_xlabel(""); ax.set_ylabel(""); 
        if ax.get_legend(): ax.get_legend().remove()
        
        short_labels = [final_mapping.get(m, m) for m in order]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=tick_fontsize)
        
        # --- EVIDENT TICKS ---
        ax.tick_params(axis='both', which='major', width=2, length=6, 
                       labelsize=tick_fontsize, bottom=True, left=True)
        
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- ROW 2: LEGEND 1 (LINES & CLASSES) ---
    ax_leg1 = fig.add_subplot(gs_bot[2, :])
    ax_leg1.axis('off')
    
    h_alpha = mlines.Line2D([], [], color=final_class_colors['Alpha'], ls=':', lw=legend_line_width, label='Alpha Class')
    h_beta  = mlines.Line2D([], [], color=final_class_colors['Beta'], ls='--', lw=legend_line_width, label='Beta Class')
    h_ref   = mlines.Line2D([], [], color='red', ls='--', lw=legend_line_width, label='Ref. Benchmark')
    
    ax_leg1.legend(handles=[h_alpha, h_beta, h_ref], loc='center', ncol=3, frameon=False, fontsize=legend_fontsize)

    if save and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"Performance_{header_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        path = output_dir / fname
        fig.savefig(path, bbox_inches='tight',dpi=IMAGE_DPI) 
        print(f"✅ Saved Graph: {path}")

    if show: plt.show()
    plt.close(fig)
    return order

def generate_journal_table(analysis_df: pd.DataFrame, 
                           metrics: list, 
                           consistent_order: list, 
                           ref_values: dict, 
                           stress_level: int = 0, 
                           font_size: int = 12,       
                           header_fontsize: int = 16, 
                           save: bool = True,
                           show: bool = False,
                           output_dir: Path = None):
    
    # set_nature_style()
    """
    Generates a Journal-style table with a Dual-Header for Per-Class performance.
    Correctly applies font sizes to BOTH tables by disabling auto-scaling.
    """
    # 0. Setup
    header_name = "TEST SET" if stress_level == 0 else f"STRESS TEST LEVEL {stress_level}"
    if stress_level is None: header_name = "AVERAGE (TEST + STRESS)"
    
    print(f"--- Table: {header_name} ---")

    # ==========================================
    # 1. DATA FILTERING
    # ==========================================
    df = analysis_df.copy()
    if stress_level is not None:
        target_df = df[df['stress_level'] == stress_level].copy()
    else:
        target_df = df[df['stress_level'] >= 0].copy()
    
    if target_df.empty:
        print("⚠️ Warning: No data found for this stress level.")
        return

    # Filter Models
    target_df = target_df[target_df['model_name'].isin(consistent_order)].copy()
    
    # Deduplicate Order
    seen = set()
    unique_order = [x for x in consistent_order if not (x in seen or seen.add(x))]
    
    # Formatting Helper
    def format_cell(val, std):
        if np.ndim(val) > 0: val = val.iloc[0]
        if np.ndim(std) > 0: std = std.iloc[0]
        if pd.isna(val): return "-"
        return f"{val:.2f} ± {std:.2f}"

    # ==========================================
    # 2. PREPARE DATA FOR TABLE A (GENERAL)
    # ==========================================
    summary_df = target_df.groupby('model_name')[metrics].agg(['mean', 'std']).reindex(unique_order)
    
    cell_text_a = []
    for model_name in summary_df.index:
        row = [model_name]
        for m in metrics:
            mean_v = summary_df.loc[model_name, (m, 'mean')]
            std_v = summary_df.loc[model_name, (m, 'std')]
            row.append(format_cell(mean_v, std_v))
        cell_text_a.append(row)
    
    # Reference Row A
    ref_row_a = ['Benchmark Ref.']
    for m in metrics:
        if ref_values and m in ref_values:
            if isinstance(ref_values[m], dict):
                val = (ref_values[m].get('alpha', 0) + ref_values[m].get('beta', 0)) / 2
            else:
                val = ref_values[m]
            ref_row_a.append(f"{val:.2f}")
        else:
            ref_row_a.append("-")
    cell_text_a.append(ref_row_a)
    
    col_labels_a = ['Model'] + [m.upper().replace('_', ' ') for m in metrics]

    # --- NEW FEATURE: Define Column Widths for Table A ---
    # Match the first column width of Table B (0.25) and distribute the rest evenly
    num_metrics = len(metrics)
    first_col_width = 0.25
    remaining_width = 1.0 - first_col_width
    
    # Prevent division by zero just in case metrics list is empty
    metric_col_width = remaining_width / num_metrics if num_metrics > 0 else 0
    col_widths_a = [first_col_width] + [metric_col_width] * num_metrics
    
    # ==========================================
    # 3. PREPARE DATA FOR TABLE B (PER-CLASS)
    # ==========================================
    pc_metrics = ['precision', 'recall', 'f1']
    col_widths_b = [0.25] + [0.125] * 6 
    
    summary_pc = pd.DataFrame()
    cell_text_b = []
    col_labels_b = [] 

    pc_cols_exist = [f'{m}_{c}' for m in pc_metrics for c in [0, 1] if f'{m}_{c}' in target_df.columns]
    
    if pc_cols_exist:
        summary_pc = target_df.groupby('model_name')[pc_cols_exist].agg(['mean', 'std']).reindex(unique_order)
        
        for model_name in summary_pc.index:
            row = [model_name]
            for m in pc_metrics:
                for c_idx in [0, 1]: 
                    col_name = f'{m}_{c_idx}'
                    if col_name in target_df.columns:
                        mean_v = summary_pc.loc[model_name, (col_name, 'mean')]
                        std_v = summary_pc.loc[model_name, (col_name, 'std')]
                        row.append(format_cell(mean_v, std_v))
                    else:
                        row.append("-")
            cell_text_b.append(row)
            
        # Reference Row B
        ref_row_b = ['Reference Benchmark']
        for m in pc_metrics:
            if m in ref_values and isinstance(ref_values[m], dict):
                ref_row_b.append(f"{ref_values[m].get('alpha', 0):.2f}")
                ref_row_b.append(f"{ref_values[m].get('beta', 0):.2f}")
            else:
                ref_row_b.extend(["-", "-"])
        cell_text_b.append(ref_row_b)
        
        col_labels_b = ['Model'] + ['Alpha', 'Beta'] * 3
    
    # ==========================================
    # 4. PLOTTING
    # ==========================================
    rows_a = len(cell_text_a)
    rows_b = len(cell_text_b)
    
    h_a = rows_a * 0.6 + 1.5 
    h_b = rows_b * 0.6 + 2.5 
    
    fig = plt.figure(figsize=(18, h_a + h_b))
    gs = fig.add_gridspec(2, 1, height_ratios=[h_a, h_b], hspace=0.3)
    
    # --- TABLE A ---
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    
    table_a = ax1.table(cellText=cell_text_a, colLabels=col_labels_a, colWidths=col_widths_a, 
                        loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    
    # --- CRITICAL FIX: FORCE FONT SIZE ON TABLE A ---
    table_a.auto_set_font_size(False) 
    table_a.set_fontsize(font_size)
    
    # Title A
    ax1.text(0.34, 1.1, f"A. General Performance ({header_name})", 
             transform=ax1.transAxes, ha='left', va='bottom', 
             fontsize=header_fontsize, weight='bold')

    for (row, col), cell in table_a.get_celld().items():
        cell.set_edgecolor('white') 
        cell.set_text_props(fontsize=font_size) 
        if row == 0:
            cell.set_text_props(weight='bold')
        if col == 0:
            cell.set_text_props(ha='left'); cell.set_x(0.01)
    
    # Lines A
    ax1.plot([0, 1], [1, 1], color='black', lw=2, transform=ax1.transAxes, clip_on=False)
    header_bottom_y = 1 - (1/(rows_a+1))
    ax1.plot([0, 1], [header_bottom_y, header_bottom_y], color='black', lw=1, transform=ax1.transAxes)
    ax1.plot([0, 1], [0, 0], color='black', lw=2, transform=ax1.transAxes)

    # --- TABLE B ---
    if cell_text_b:
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        table_b = ax2.table(cellText=cell_text_b, colLabels=col_labels_b, 
                            colWidths=col_widths_b,
                            loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
        
        # --- CRITICAL FIX: FORCE FONT SIZE ON TABLE B ---
        table_b.auto_set_font_size(False)
        table_b.set_fontsize(font_size)
        
        # Title B
        ax2.text(0.34, 1.15, f"B. Per-Class Performance ({header_name})", 
                 transform=ax2.transAxes, ha='left', va='bottom', 
                 fontsize=header_fontsize, weight='bold')

        for (row, col), cell in table_b.get_celld().items():
            cell.set_edgecolor('white')
            if row == 0: cell.set_text_props(weight='bold')
            if col == 0: cell.set_text_props(ha='left'); cell.set_x(0.01)

        # Super Headers
        y_text = 1.02; y_line = 1.0
        
        # Precision
        ax2.text(0.375, y_text, "PRECISION", ha='center', va='bottom', weight='bold', 
                 fontsize=font_size, transform=ax2.transAxes)
        ax2.plot([0.26, 0.49], [y_line, y_line], color='black', lw=1, transform=ax2.transAxes)

        # Recall
        ax2.text(0.625, y_text, "RECALL", ha='center', va='bottom', weight='bold', 
                 fontsize=font_size, transform=ax2.transAxes)
        ax2.plot([0.51, 0.74], [y_line, y_line], color='black', lw=1, transform=ax2.transAxes)

        # F1
        ax2.text(0.875, y_text, "F1 SCORE", ha='center', va='bottom', weight='bold', 
                 fontsize=font_size, transform=ax2.transAxes)
        ax2.plot([0.76, 0.99], [y_line, y_line], color='black', lw=1, transform=ax2.transAxes)

        # Lines B
        ax2.plot([0, 1], [1.12, 1.12], color='black', lw=2, transform=ax2.transAxes, clip_on=False)
        y_sub_bottom = 1 - (1 / (rows_b + 1))
        ax2.plot([0, 1], [y_sub_bottom, y_sub_bottom], color='black', lw=1, transform=ax2.transAxes)
        ax2.plot([0, 1], [0, 0], color='black', lw=2, transform=ax2.transAxes)

    # 5. Output
    if save and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"Performance_Table_{header_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        path = output_dir / fname
        fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=IMAGE_DPI)
        print(f"✅ Saved Table: {path}")

    if show:
        plt.show()
    
    plt.close(fig)



# ======================================================================================
# Plotting Function for Comparison Visualization
# ======================================================================================

#_______________________________________________________________
# Helpers 

def perform_pairwise_mannwhitneyu(df, x_col, y_col):
    """
    Calculates Mann-Whitney U test between the unique groups in x_col.
    Returns: [(group1, group2, u_stat, p_val)]
    """
    groups = df[x_col].unique()
    if len(groups) != 2: return []
    g1, g2 = groups[0], groups[1]
    data1 = df[df[x_col] == g1][y_col]
    data2 = df[df[x_col] == g2][y_col]
    
    if len(data1) == 0 or len(data2) == 0: return []

    u_stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
    return [(g1, g2, u_stat, p_val)]

def add_stat_annotation(ax, df, x_col, y_col, order, stat_results, font_size=12):
    """
    Draws Nature-style significance brackets.
    """
    if not stat_results: return
    _, _, _, p = stat_results[0]
    
    if p >= 0.05: text = 'ns'
    elif p >= 0.01: text = '*'
    elif p >= 0.001: text = '**'
    elif p >= 0.0001: text = '***'
    else: text = '****'
    
    y_max = df[y_col].max()
    y_range = df[y_col].max() - df[y_col].min()
    if y_range == 0: y_range = 1.0
    
    # Bracket positioning
    y_line = y_max + 0.05 * y_range 
    y_text = y_line + 0.01 * y_range
    
    # Coordinates for models (assuming 0 and 1 on x-axis)
    x1, x2 = 0, 1
    
    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y_line, y_line + 0.02*y_range, y_line + 0.02*y_range, y_line], 
            lw=1.0, c='black')
    
    # Draw text
    ax.text((x1 + x2) * 0.5, y_text, text, ha='center', va='bottom', 
            fontsize=font_size, color='black')

#_______________________________________________________________
# Plotting

def plot_model_comparison(
    master_df,
    models_to_compare,
    analysis_metrics=None,
    stress_level=0,  # Adapted to your setup
    plot_palette=None,
    wspace = 0.3,
    hspace = 0.4,
    
    # Font Sizes Controls
    title_fontsize=16,
    axis_fontsize=12,
    stat_fontsize=12,
    letter_fontsize=20,
    save=True,
    show=True,
    output_dir=None
):
    set_nature_style()
    
    # Defaults
    if analysis_metrics is None: analysis_metrics = ANALYSIS_METRICS 
    if plot_palette is None: plot_palette = MODEL_COLORS # Use Global
    if output_dir is None: output_dir = Path(".")

    # Filter Data (Using stress_level instead of data_split)
    comparison_df = master_df[
        (master_df['model_name'].isin(models_to_compare)) &
        (master_df['stress_level'] == stress_level)
    ].copy()
    
    if comparison_df.empty:
        print(f"⚠️ Warning: No data found for models {models_to_compare} at stress level {stress_level}")
        return

    n_metrics = len(analysis_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), squeeze=False)
    
    # Props
    box_props = dict(facecolor='none', edgecolor='black', linewidth=1.2)
    whisker_props = dict(linewidth=1.2, color='black')
    cap_props = dict(linewidth=1.2, color='black')
    median_props = dict(linewidth=1.5, color='black')

    for i, metric in enumerate(analysis_metrics):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        # Letter Label
        letter = chr(ord('A') + i)
        ax.text(-0.1, 1.1, f"{letter}.", transform=ax.transAxes,
                fontsize=letter_fontsize, fontweight='bold', va='top')

        # 1. Stripplot (Points)
        sns.stripplot(
            data=comparison_df, x='model_name', y=metric,
            hue='model_name', legend=False,
            order=models_to_compare, palette=plot_palette,
            ax=ax, size=8, alpha=1.0, jitter=True, zorder=0
        )
        
        # 2. Boxplot (Distribution)
        sns.boxplot(
            data=comparison_df, x='model_name', y=metric,
            order=models_to_compare, ax=ax,
            width=0.4, showfliers=False,
            boxprops=box_props, whiskerprops=whisker_props,
            capprops=cap_props, medianprops=median_props, zorder=10
        )

        # 3. Stats Annotation
        stat_results = perform_pairwise_mannwhitneyu(comparison_df, 'model_name', metric)
        if stat_results:
             add_stat_annotation(ax, comparison_df, 'model_name', metric, models_to_compare, stat_results, font_size=stat_fontsize)

        # Styling
        ax.set_title(metric.upper().replace('_', ' '), fontsize=title_fontsize, weight='bold', pad=10)
        ax.set_ylabel('') 
        ax.set_xlabel('')
        
        # X Ticks
        ax.set_xticks(range(len(models_to_compare)))
        ax.set_xticklabels(models_to_compare, fontsize=axis_fontsize)
        
        # Y Ticks
        ax.tick_params(axis='y', labelsize=axis_fontsize)
        
        # Grid (Horizontal only)
        ax.grid(axis='y', linestyle=':', alpha=0.4, color='grey')
        
        # Y Limits Adjustment for Bracket space
        y_min, y_max = comparison_df[metric].min(), comparison_df[metric].max()
        margin = (y_max - y_min) * 0.3 if (y_max - y_min) > 0 else 0.1
        ax.set_ylim(max(0, y_min - margin/2), y_max + margin)

    # Turn off unused axes
    for j in range(i + 1, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis('off')
    
    plt.subplots_adjust( 
        left=0.05, 
        right=0.95, 
        top=0.92, 
        bottom=0.15,
        wspace=wspace, 
        hspace=hspace
    )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plot_palette.get(m, 'k'), 
                   markersize=12, label=m)
        for m in models_to_compare
    ]

    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(models_to_compare),
        frameon=False,
        fontsize=title_fontsize, 
        bbox_to_anchor=(0.5, 0.0)
    )

    if save and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / f"Comparison_{models_to_compare[0]}_vs_{models_to_compare[1]}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches='tight', dpi=IMAGE_DPI)
        print(f"✅ Saved Comparison Plot: {path}")

    if show: plt.show()
    plt.close(fig)

def plot_model_comparison_table(
    master_df,
    models_to_compare,
    analysis_metrics=None,
    stress_level=0,
    # Fonts
    title_fontsize=16,
    header_fontsize=12,
    cell_fontsize=12,
    save=True,
    show=True,
    output_dir=None
):
    
    set_nature_style()


    if analysis_metrics is None: analysis_metrics = ANALYSIS_METRICS
    if output_dir is None: output_dir = Path(".")
        
    # Filter Data
    comparison_df = master_df[
        (master_df['model_name'].isin(models_to_compare)) &
        (master_df['stress_level'] == stress_level)
    ].copy()

    # --- PART 1: GENERAL METRICS TABLE ---
    summary = comparison_df.groupby('model_name')[analysis_metrics].agg(['mean', 'std'])
    
    cell_text_a = []
    for metric in analysis_metrics:
        row = []
        for model in models_to_compare:
            m = summary.loc[model, (metric, 'mean')]
            s = summary.loc[model, (metric, 'std')]
            row.append(f"{m:.2f} ± {s:.2f}")

        stats = perform_pairwise_mannwhitneyu(comparison_df, 'model_name', metric)
        if stats:
            _, _, u, p = stats[0]
            p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
            row += [f"{u:.1f}", p_str]
        else:
            row += ["–", "–"]
        cell_text_a.append(row)

    col_labels = models_to_compare + ["U-stat", "p-value"]
    row_labels_a = [m.upper().replace('_', ' ') for m in analysis_metrics]

    # --- PART 2: PER-CLASS METRICS TABLE ---
    pc_metrics = ['precision', 'recall', 'f1']
    cell_text_b = []
    row_labels_b = []
    
    for metric_base in pc_metrics:
        for cls in [0, 1]: # Alpha/Beta
            col_name = f"{metric_base}_{cls}"
            if col_name not in comparison_df.columns: continue
            
            row = []
            cls_name = "Alpha" if cls == 0 else "Beta"
            row_labels_b.append(f"{metric_base.upper()} ({cls_name})")
            
            summary_pc = comparison_df.groupby('model_name')[col_name].agg(['mean', 'std'])
            
            for model in models_to_compare:
                m = summary_pc.loc[model, 'mean']
                s = summary_pc.loc[model, 'std']
                row.append(f"{m:.2f} ± {s:.2f}")
            
            stats = perform_pairwise_mannwhitneyu(comparison_df, 'model_name', col_name)
            if stats:
                _, _, u, p = stats[0]
                p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
                row += [f"{u:.1f}", p_str]
            else:
                row += ["–", "–"]
            cell_text_b.append(row)

    # --- PLOTTING ---
    n_rows_a = len(cell_text_a)
    n_rows_b = len(cell_text_b)
    
    h_a = (n_rows_a + 2) * 0.6
    h_b = (n_rows_b + 2) * 0.6
    
    fig = plt.figure(figsize=(14, h_a + h_b))
    gs = fig.add_gridspec(2, 1, height_ratios=[h_a, h_b], hspace=0.3)
    ax1 = fig.add_subplot(gs[0]); ax1.axis('off')
    ax2 = fig.add_subplot(gs[1]); ax2.axis('off')

    # --- STYLE FUNCTION (Nature Journal) ---
    def render_clean_table(ax, cell_text, row_labels, col_labels, title):
        table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, 
                         loc='center', cellLoc='center', bbox=[0.15, 0, 0.85, 1])
        
        # Explicit Font Control
        table.auto_set_font_size(False)
        table.set_fontsize(cell_fontsize)
        
        # Title
        ax.text(0.5, 1.02, title, transform=ax.transAxes, ha='center', va='bottom', 
                fontsize=title_fontsize, weight='bold')

        # Styling
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('white') # No inner grid
            cell.set_height(0.1)
            
            # Header Row
            if row == 0:
                cell.set_text_props(weight='bold', fontsize=header_fontsize)
                # Bottom Line for Header
                cell.set_edgecolor('black')
                cell.set_linewidth(0) # We draw manual lines
            
            # Row Labels (Col -1)
            if col == -1:
                cell.set_text_props(weight='bold', fontsize=header_fontsize, ha='right')

        # Manual Lines
        ax.plot([0.15, 1.0], [1.0, 1.0], color='black', lw=2, transform=ax.transAxes) # Top
        y_hdr = 1.0 - (1.0/(len(cell_text)+1))
        ax.plot([0.15, 1.0], [y_hdr, y_hdr], color='black', lw=1, transform=ax.transAxes) # Header Bottom
        ax.plot([0.15, 1.0], [0.0, 0.0], color='black', lw=2, transform=ax.transAxes) # Bottom
        
        return table

    # Render
    render_clean_table(ax1, cell_text_a, row_labels_a, col_labels, "General Performance (Pairwise)")
    render_clean_table(ax2, cell_text_b, row_labels_b, col_labels, "Per-Class Performance (Pairwise)")

    if save and output_dir:
        path = output_dir / f"Comparison_{models_to_compare[0]}_vs_{models_to_compare[1]}_Table.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches='tight', facecolor='white', dpi=IMAGE_DPI)
        print(f"✅ Saved Comparison Table: {path}")

    if show: plt.show()
    plt.close(fig)



# ======================================================================================
# Plotting Function for Drop in Performances Evaluation
# ======================================================================================

def plot_robustness_drops_graphs(
    master_df, 
    ANALYSIS_METRICS, 
    COLOR_CODEX, 
    OUTPUT_DIR, 
    name_mapping=None, 
    title_fontsize=16,
    axis_fontsize=12,
    ball_fontsize=14,
    legend_fontsize=16,
    h_pad_= 3.0,
    w_pad_= 2.0,
    save=True, 
    show=True
):
    """
    Generates Nature-style Bar Charts showing performance drops (Test - Stress).
    """
    # 1. STYLE ENFORCEMENT
    set_nature_style()

    print("\n" + "="*60)
    print("Generating Robustness Graphs (Performance Drops)")
    print("="*60)
    
    # 2. DATA PREPARATION
    all_metrics_drops = []
    
    # Normalize Columns
    df = master_df.copy()
    df.columns = [c.lower() if c in ['AUC', 'PR_AUC'] else c for c in df.columns]
    
    for metric in ANALYSIS_METRICS:
        metric = metric.lower() # Ensure match
        drop_list = []
        
        # Calculate drops per seed to preserve variance
        if 'seed' in df.columns:
            seeds = df['seed'].unique()
        else:
            seeds = [0] # Handle case with no seeds
            
        for seed in seeds:
            if 'seed' in df.columns:
                seed_df = df[df['seed'] == seed]
            else:
                seed_df = df
            
            # Baseline (Level 0) vs Stress (Level > 0)
            # Assuming we want the average drop across all stress levels vs baseline
            test_df = seed_df[seed_df['stress_level'] == 0][['model_name', metric]]
            stress_df = seed_df[seed_df['stress_level'] > 0][['model_name', metric]]
            
            if test_df.empty or stress_df.empty: continue
            
            # We average the stress results if multiple levels exist, or take specific logic?
            # Standard approach: Avg(Test) - Avg(Stress) per model
            test_agg = test_df.groupby('model_name')[metric].mean().reset_index().rename(columns={metric: 'score_test'})
            stress_agg = stress_df.groupby('model_name')[metric].mean().reset_index().rename(columns={metric: 'score_stress'})
            
            merged = pd.merge(test_agg, stress_agg, on='model_name')
            
            merged['drop'] = merged['score_test'] - merged['score_stress']
            merged['metric'] = metric
            drop_list.append(merged)
            
        if drop_list:
            all_metrics_drops.append(pd.concat(drop_list))

    if not all_metrics_drops:
        print("⚠️ No data available for robustness plotting.")
        return

    full_drop_df = pd.concat(all_metrics_drops)

    # Apply Name Mapping
    if name_mapping:
        full_drop_df['plot_name'] = full_drop_df['model_name'].map(name_mapping).fillna(full_drop_df['model_name'])
        plot_palette = {name_mapping.get(k, k): v for k, v in COLOR_CODEX.items()}
    else:
        full_drop_df['plot_name'] = full_drop_df['model_name']
        plot_palette = COLOR_CODEX

    # Sort Order (Low drop is good, so we might want to sort by drop ascending or descending)
    # Sorting by ascending drop (smallest drop first) based on first metric
    first_metric = ANALYSIS_METRICS[0].lower()
    sorter = full_drop_df[full_drop_df['metric'] == first_metric] \
              .groupby('plot_name')['drop'].mean() \
              .sort_values(ascending=True).index.tolist()

    # 3. PLOTTING SETUP
    n_metrics = len(ANALYSIS_METRICS)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows), squeeze=False)
    
    for i, metric in enumerate(ANALYSIS_METRICS):
        metric = metric.lower()
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        
        plot_data = full_drop_df[full_drop_df['metric'] == metric]
        
        # Letter Label
        letter = chr(ord('A') + i)
        ax.text(-0.1, 1.1, f"{letter}.", transform=ax.transAxes, 
                fontsize=20, fontweight='bold', va='top')
        
        # Bar Plot
        sns.barplot(
            data=plot_data,
            y='plot_name',
            x='drop',
            order=sorter,
            palette=plot_palette,
            hue='plot_name',
            ax=ax,
            dodge=False,
            errorbar='sd',
            err_kws={'linewidth': 1.5, 'color': 'black'},
            capsize=0.1,
            legend=False,
            edgecolor='black',
            linewidth=1.0
        )
        
        ax.set_title(f"Performance Drop ({metric.upper()})", fontsize=title_fontsize, weight='bold', pad=10)
        ax.set_xlabel("Mean Drop (Lower is Better)", fontsize=axis_fontsize)
        ax.set_ylabel("") 
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid
        ax.grid(axis='x', linestyle=':', alpha=0.5, color='grey')
        
        # Ticks
        ax.tick_params(axis='both', labelsize=axis_fontsize) 
        ax.axvline(0, color='black', linewidth=1.0, linestyle='-')

    # Hide empty subplots
    for j in range(i + 1, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis('off')

    # Legend
    if name_mapping:
        legend_handles = []
        unique_models = full_drop_df['model_name'].unique()
        for real_name in unique_models:
            if real_name in COLOR_CODEX:
                short = name_mapping.get(real_name, real_name)
                color = COLOR_CODEX[real_name]
                label_text = f"{short} : {real_name}"
                handle = Line2D([0], [0], marker='o', color='w', label=label_text,
                                markerfacecolor=color, markersize=ball_fontsize/1.5) # scaled slightly
                legend_handles.append(handle)
        
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.02),
                ncol=4,
                frameon=False,
                fontsize=legend_fontsize
            )

    # 4. LAYOUT ADJUSTMENT
    rect_bottom = 0.1 if name_mapping else 0.05
    plt.tight_layout(rect=[0, rect_bottom, 1, 0.98], h_pad=h_pad_, w_pad=w_pad_)
    
    # 5. SAVE
    if save and OUTPUT_DIR:
        save_path = Path(OUTPUT_DIR) / "Robustness_Drops_Graphs.png"
        print(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=IMAGE_DPI, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved Graphs: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

#_______________________________________________________________
# 2. ROBUSTNESS TABLE 

def smart_fmt(x):
    if abs(x) >= 10:
        return f"{x:.0f}"
    elif abs(x) >= 1:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"

def plot_robustness_drops_table(
    master_df, 
    ANALYSIS_METRICS, 
    OUTPUT_DIR,
    header_fontsize=14,    
    row_label_fontsize=12,
    cell_fontsize=12, 
    title_fontsize=26,
    title_y= 1.1,
    offset = 0.005,
    ROW_HEIGHT = 0.08,
    scale_factor=1e3,      
    scale_power=-3,        
    save=True, 
    show=True
):

    """
    Generates a Nature-style Table summary of performance drops.
    - Model Names are strictly in Column 0.
    - Bolds the Best (Lowest Drop) result per metric.
    - Italics the Second Best result per metric.
    """
    # 1. STYLE ENFORCEMENT
    set_nature_style()

    print("\n" + "="*60)
    print("Generating Robustness Table (Adjusted Header Line)")
    print("="*60)
    
    # 2. DATA PREPARATION
    df = master_df.copy()
    df.columns = [c.lower() if c in ['AUC', 'PR_AUC'] else c for c in df.columns]
    
    model_stats = {} 
    raw_means = {}   
    for m in ANALYSIS_METRICS: raw_means[m.lower()] = {}

    for metric in ANALYSIS_METRICS:
        metric = metric.lower()
        drop_list = []
        
        if 'seed' in df.columns: seeds = df['seed'].unique()
        else: seeds = [0]

        for seed in seeds:
            seed_df = df[df['seed'] == seed] if 'seed' in df.columns else df
            test_df = seed_df[seed_df['stress_level'] == 0][['model_name', metric]]
            stress_df = seed_df[seed_df['stress_level'] > 0][['model_name', metric]]
            
            if test_df.empty or stress_df.empty: continue
            
            test_agg = test_df.groupby('model_name')[metric].mean().reset_index().rename(columns={metric: 'test'})
            stress_agg = stress_df.groupby('model_name')[metric].mean().reset_index().rename(columns={metric: 'stress'})

            merged = pd.merge(test_agg, stress_agg, on='model_name')
            merged['drop'] = merged['test'] - merged['stress']
            drop_list.append(merged)
            
        if drop_list:
            full_metric_df = pd.concat(drop_list)
            agg = full_metric_df.groupby('model_name')['drop'].agg(['mean', 'std'])
            for model in agg.index:
                if model not in model_stats: model_stats[model] = {}
                m_val = agg.loc[model, 'mean']
                s_val = agg.loc[model, 'std']
                raw_means[metric][model] = m_val
                m_scaled = m_val * scale_factor
                s_scaled = s_val * scale_factor
                model_stats[model][metric] = f"{smart_fmt(m_scaled)} ± {smart_fmt(s_scaled)}"


    table_df = pd.DataFrame.from_dict(model_stats, orient='index')
    
    # Reorder columns & Sort
    norm_metrics = [m.lower() for m in ANALYSIS_METRICS]
    table_df = table_df[norm_metrics] 
    
    first_metric = norm_metrics[0]
    def get_mean(x):
        try: return float(x.split(' ± ')[0])
        except: return 999.0
    table_df['sort_val'] = table_df[first_metric].apply(get_mean)
    table_df = table_df.sort_values('sort_val', ascending=True).drop(columns=['sort_val'])

    # 3. RANKING LOGIC (Best/Second Best)
    rankings = {}
    for metric in norm_metrics:
        sorted_models = sorted(raw_means[metric].items(), key=lambda x: x[1])
        best_model = sorted_models[0][0] if len(sorted_models) > 0 else None
        second_model = sorted_models[1][0] if len(sorted_models) > 1 else None
        rankings[metric] = {'best': best_model, 'second': second_model}

    # 4. LAYOUT CONSTANTS
    row_labels = table_df.index.tolist()
    # Explicitly include "Model" as the first column header
    metric_labels = [ f"{m.upper().replace('_', ' ')} \n(×10^{scale_power})" for m in norm_metrics ]
    col_labels = ["Model"] + metric_labels 
    
    n_data_rows = len(row_labels)
    n_total_rows = n_data_rows + 1 
     
    
    # 5. PLOTTING
    fig, ax = plt.subplots(figsize=(20, n_total_rows * 0.8 + 1)) 
    ax.axis('off')
    
    # Column Widths: Give "Model" (Col 0) more space, split rest evenly
    model_col_width = 0.17
    data_col_width = (0.95 - model_col_width) / len(metric_labels)
    col_widths = [model_col_width] + [data_col_width] * len(metric_labels)

    # Prepare cell text: [Model Name, Data, Data...]
    cell_text = []
    for model in row_labels:
        row_data = table_df.loc[model].tolist()
        cell_text.append([model] + row_data)

    # Create Table
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels, # No rowLabels arg needed now
        colWidths=col_widths,
        loc='upper center', 
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(cell_fontsize) 

    # 6. TITLE & HEADER LINES
    ax.text(0.5, title_y, "Performance Degradation (Test Set vs. Stress Conditions)", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=title_fontsize, weight='bold')

    # 7. STYLING & HIGHLIGHTING
    for (row, col), cell in table.get_celld().items():
        cell.set_height(ROW_HEIGHT)
        cell.set_edgecolor('white') 
        
        # --- A. HEADER ROW ---
        if row == 0:
            cell.set_text_props(weight='bold', fontsize=header_fontsize)
            # Left align "Model" header
            if col == 0: 
                cell.set_text_props(ha='left')
                cell.set_x(0.02) # Indent

        # --- B. DATA ROWS ---
        else:
            current_model = row_labels[row-1]
            
            # Column 0: Model Name
            if col == 0:
                cell.set_text_props(
                    weight='bold', 
                    fontsize=row_label_fontsize, 
                    ha='left',
                    color='black'
                )
                cell.set_x(0.02) # Indent slightly
            
            # Columns > 0: Metrics
            else:
                metric_idx = col - 1
                current_metric = norm_metrics[metric_idx]
                
                # Apply Highlighting
                if current_model == rankings[current_metric]['best']:
                    cell.set_text_props(weight='bold', fontsize=cell_fontsize)
                elif current_model == rankings[current_metric]['second']:
                    cell.set_text_props(style='italic', fontsize=cell_fontsize)
                else:
                    cell.set_text_props(fontsize=cell_fontsize)

    # Horizontal Lines
    line_x_start = 0.5 - (sum(col_widths) / 2)
    line_x_end = 0.5 + (sum(col_widths) / 2)
    
    y_top = 1.0
    
    # --- ADJUSTMENT HERE ---
    # We subtract a small offset (e.g., 0.005) from the standard header bottom calculation
    # Standard was: 1.0 - ROW_HEIGHT
    
    y_hdr_bot = 1.0 - ROW_HEIGHT - offset 
    
    y_btm = 1.0 - (n_total_rows * ROW_HEIGHT)

    ax.plot([line_x_start, line_x_end], [y_top, y_top], color='black', linewidth=2.0, transform=ax.transAxes)
    ax.plot([line_x_start, line_x_end], [y_hdr_bot, y_hdr_bot], color='black', linewidth=1.0, transform=ax.transAxes)
    ax.plot([line_x_start, line_x_end], [y_btm, y_btm], color='black', linewidth=2.0, transform=ax.transAxes)

    # 8. SAVE
    if save and OUTPUT_DIR:
        save_path = Path(OUTPUT_DIR) / "Robustness_Drops_Table.png"
        print(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=IMAGE_DPI)
        print(f"✅ Saved Table: {save_path}")

    if show:
        plt.show()        
    plt.close(fig)


#_______________________________________________________________
# One graph for all mperformance metrics in all stress levels in one graph

def plot_separated_model_analysis_final(
    master_df: pd.DataFrame, 
    model_name: str, 
    metrics_to_plot: list, 
    ref_values: dict = None,                         
    save: bool = False,
    show: bool = True,
    point: bool = False,
    offset: float = 0.05,
    eq_margin_multiplier: float = 0.5, # Rigorous equivalence multiplier
    # Font Sizes
    title_fontsize: int = 22,
    axis_label_fontsize: int = 16,
    tick_fontsize: int = 14,
    legend_fontsize: int = 14,
    table_header_fontsize: int = 14,
    table_cell_fontsize: int = 14,
    OUTPUT_DIR: Path = Path('Stress_Test_Results')
):
    """
    Generates TWO separate figures in Nature-Journal Style.
    Plot B uses a rigorous Equivalence Forest Plot with dynamically 
    calculated margins based on baseline variance, oriented vertically 
    (metrics on Y-axis, drops on X-axis).
    """
    print(f"\n" + "="*60)
    print(f"--- Processing Analysis for Model: {model_name} ---")
    print("="*60)
    
    # 2. DATA PREPARATION
    model_df = master_df[
        (master_df['model_name'] == model_name) & 
        (master_df['data_split'].isin(['Test', 'Stress']))
    ].copy()

    if model_df.empty: 
        print(f"⚠️ Warning: No data found for {model_name}")
        return
    if model_df['stress_level'].max() == 0: 
        print("⚠️ Warning: No stress data found.")
        return

    # --- Prep Boxplot Data ---
    melted_df = pd.melt(model_df, id_vars=['seed', 'stress_level'], value_vars=metrics_to_plot,
                        var_name='metric', value_name='score')

    # --- Prep Forest Plot Data ---
    summary_data = []
    all_drops_list = []
    
    for metric in metrics_to_plot:
        baseline = model_df[model_df['stress_level'] == 0][metric].dropna()
        stress = model_df[model_df['stress_level'] > 0][metric].dropna()
        
        base_seed = model_df[model_df['stress_level'] == 0][['seed', metric]].set_index('seed')
        stress_seed = model_df[model_df['stress_level'] > 0][['seed', metric]].set_index('seed')
        merged = base_seed.join(stress_seed, lsuffix='_base', rsuffix='_stress').dropna()
        
        drop_dist = merged[f'{metric}_base'] - merged[f'{metric}_stress']
        temp_drop_df = pd.DataFrame({'metric': metric, 'drop_val': drop_dist.values})
        all_drops_list.append(temp_drop_df)
        
        # Rigorous Equivalence Margin Calculation
        baseline_std = baseline.std() if not baseline.empty else 0
        dynamic_eq_margin = eq_margin_multiplier * baseline_std

        # Calculate 95% Confidence Interval for the mean drop
        n_seeds = len(drop_dist)
        mean_drop = drop_dist.mean()
        std_drop = drop_dist.std()
        sem_drop = std_drop / np.sqrt(n_seeds) if n_seeds > 0 else 0
        ci_95 = t.ppf(0.975, n_seeds-1) * sem_drop if n_seeds > 1 else 0

        u_stat, p_val = 'N/A', 1.0
        if not baseline.empty and not stress.empty:
            u_stat, p_val = mannwhitneyu(baseline, stress, alternative='two-sided')
        
        summary_data.append({
            'metric': metric, 
            'mean_drop': mean_drop, 
            'std_drop': std_drop, 
            'ci_95': ci_95,
            'eq_margin': dynamic_eq_margin,
            'u_statistic': u_stat, 
            'p_value': p_val
        })
        
    summary_stats_drops = pd.DataFrame(summary_data).set_index('metric').reindex(metrics_to_plot)
    raw_drops_df = pd.concat(all_drops_list) if all_drops_list else pd.DataFrame()
    metric_labels_upper = [m.upper().replace('_', ' ') for m in metrics_to_plot]

    # ==========================================================================
    # 3. FIGURE 1: GRAPHS
    # ==========================================================================
    fig_graphs, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(20, 12), gridspec_kw={'width_ratios': [1.2, 1]})
    fig_graphs.patch.set_facecolor('white')
    
    # Palette setup
    n_levels = model_df['stress_level'].nunique()
    stress_colors = sns.color_palette("Blues", n_colors=n_levels + 2)[2:]
    unique_levels = sorted(model_df['stress_level'].unique())
    custom_palette = {}
    red_idx = 0
    for lvl in unique_levels:
        if lvl == 0: custom_palette[lvl] = '#d9d9d9'
        else:
            custom_palette[lvl] = stress_colors[red_idx]
            red_idx += 1
            
    # --- PLOT A (Unchanged) ---
    ax_box.text(-0.08, 1.05, 'A.', transform=ax_box.transAxes, fontsize=24, fontweight='bold', va='top')
    sns.boxplot(data=melted_df, x='metric', y='score', hue='stress_level',
                order=metrics_to_plot, palette=custom_palette, ax=ax_box,
                linewidth=1.2, fliersize=3,
                boxprops=dict(edgecolor='black', alpha=0.9), 
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color='black'))
    
    for i in range(len(metrics_to_plot) - 1):
        ax_box.axvline(i + 0.5, color='grey', linestyle='-', linewidth=0.5, alpha=0.3)

    if ref_values:
        for i, metric in enumerate(metrics_to_plot):
            if metric in ref_values:
                ref_val = np.mean(list(ref_values[metric].values())) if isinstance(ref_values[metric], dict) else ref_values[metric]
                ax_box.hlines(y=ref_val, xmin=i - 0.4, xmax=i + 0.4, color='red', linestyle='--', linewidth=2.5, zorder=10)

    ax_box.set_title(f"Distribution of Scores", fontsize=title_fontsize, weight='bold', pad=15)
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Absolute Score", fontsize=axis_label_fontsize)
    ax_box.set_xticklabels(metric_labels_upper, fontsize=tick_fontsize, fontweight='bold', rotation=30, ha='right')
    ax_box.tick_params(axis='y', labelsize=tick_fontsize, left=True, width=1.5, length=5)
    ax_box.grid(axis='y', linestyle=':', alpha=0.5, color='grey')
    ax_box.legend(title='Stress Level', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_levels, frameon=False, fontsize=legend_fontsize, title_fontsize=legend_fontsize)

    # --- PLOT B (Vertical Forest Plot) ---
    ax_bar.text(-0.08, 1.05, 'B.', transform=ax_bar.transAxes, fontsize=24, fontweight='bold', va='top')
    y_pos = np.arange(len(summary_stats_drops)) # Metrics are now on the Y-axis
    means = summary_stats_drops['mean_drop']
    errors_ci = summary_stats_drops['ci_95']
    margins = summary_stats_drops['eq_margin']

    # Draw individual Equivalence Margins for each metric horizontally across the specific y-coordinate
    for i, margin in enumerate(margins):
        ax_bar.fill_betweenx([i - 0.4, i + 0.4], -margin, margin, color='green', alpha=0.15, zorder=0)

    # Draw the strict zero line (vertical)
    ax_bar.axvline(0, color='black', linewidth=1.2, linestyle='--', zorder=1)

    # Plot the raw scatter points (optional)
    if not raw_drops_df.empty and point:
        metric_map = {m: i for i, m in enumerate(metrics_to_plot)}
        raw_drops_df['y_idx'] = raw_drops_df['metric'].map(metric_map)
        # Note: orient='h' and swap x/y for horizontal stripplot
        sns.stripplot(data=raw_drops_df, y='y_idx', x='drop_val', color='grey', alpha=0.4, jitter=0.15, size=4, ax=ax_bar, zorder=2, orient='h')

    # Plot the Mean Drop and 95% CI (Horizontal error bars)
    ax_bar.errorbar(means, y_pos, xerr=errors_ci, fmt='o', color='black', 
                    markersize=9, capsize=6, capthick=2, elinewidth=2, zorder=3, label='Mean Drop (95% CI)')

    # Add significance stars mapping to the right side of the error bars
    for i, p_val in enumerate(summary_stats_drops['p_value']):
        x_visual_right = means.iloc[i] + errors_ci.iloc[i]
        x_pos_star = x_visual_right + (abs(x_visual_right) * 0.05) + 0.002
        if p_val < 0.05:
            symbol = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ax_bar.text(x_pos_star, y_pos[i], symbol, ha='left', va='center', color='black', fontsize=18, weight='bold')

    # Custom legend for the dynamic margins
    eq_patch = Patch(color='green', alpha=0.15, label=r'Equivalence Margin ($0.5 \times \sigma_{baseline}$)')
    handles, labels = ax_bar.get_legend_handles_labels()
    handles.append(eq_patch)

    ax_bar.set_title(f"Mean Performance Drop", fontsize=title_fontsize, weight='bold', pad=15)
    ax_bar.set_ylabel("") # No label needed, it's obvious from the metric names
    ax_bar.set_xlabel("Drop (Baseline - Stress)", fontsize=axis_label_fontsize)
    
    # Configure Y-axis for metrics
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(metric_labels_upper, fontsize=tick_fontsize, fontweight='bold')
    ax_bar.invert_yaxis() # Important: This makes the first metric (AUC) appear at the top!
    
    ax_bar.tick_params(axis='x', labelsize=tick_fontsize)
    ax_bar.grid(axis='x', linestyle=':', alpha=0.5, color='grey') # Grid lines are vertical now
    # Aligns the legend perfectly with Plot A's legend underneath the graph
    ax_bar.legend(
        handles=handles, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), # Matches the exact Y-height of Plot A's legend
        ncol=1,                      # Stacks the two long labels neatly
        frameon=False, 
        fontsize=legend_fontsize
    )

    plt.tight_layout(pad=3.0, rect=[0, 0.05, 1, 1]) 

    # ==========================================================================
    # 4. FIGURE 2: TABLE (Aligned Logic)
    # ==========================================================================
    summary_stats_abs = model_df.groupby('stress_level')[metrics_to_plot].agg(['mean', 'std'])
    stress_levels_sorted = sorted(model_df['stress_level'].unique())
    
    table_rows = []
    row_headers = []
    
    # 1. Absolute Scores: Show Mean ± SD to indicate raw model variance across seeds
    for lvl in stress_levels_sorted:
        row_data = [f"{summary_stats_abs.loc[lvl, (m, 'mean')]:.3f} ± {summary_stats_abs.loc[lvl, (m, 'std')]:.3f}" for m in metrics_to_plot]
        table_rows.append(row_data)
        row_headers.append(f"Level {lvl} ")
        
    # 2. Mean Drop: Show Mean ± 95% CI to match the Forest Plot (Plot B)
    drop_row = summary_stats_drops.apply(lambda r: f"{r['mean_drop']:.3f} ± {r['ci_95']:.3f}", axis=1).tolist()
    table_rows.append(drop_row)
    row_headers.append("Mean Drop")
    
    # 3. P-Value: Unchanged
    pval_row = summary_stats_drops['p_value'].apply(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001").tolist()
    table_rows.append(pval_row)
    row_headers.append("P-Value")

    # Layout Calcs
    n_cols = len(metrics_to_plot)
    n_data_rows = len(table_rows)
    ROW_HEIGHT = 0.12 
    
    fig_table, ax_table = plt.subplots(figsize=(18, n_data_rows * 0.8 + 2))
    ax_table.axis('off')
    
    y_start = 0.9 - offset
    total_table_height = ROW_HEIGHT * (n_data_rows + 1)
    y_end = y_start - total_table_height
    bbox = [0.025, y_end, 0.95, total_table_height]
    
    col_widths = [0.15] + [(0.95 - 0.15) / n_cols] * n_cols 
    cell_text_final = [[row_headers[i]] + r for i, r in enumerate(table_rows)]
    col_labels = ["Statistic"] + metric_labels_upper
    
    table = ax_table.table(
        cellText=cell_text_final, colLabels=col_labels, colWidths=col_widths,
        loc='center', cellLoc='center', bbox=bbox
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_cell_fontsize)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if row == 0:
            cell.set_text_props(weight='bold', fontsize=table_header_fontsize)
            if col == 0: cell.set_text_props(ha='left'); cell.set_x(0.04)
        else:
            if col == 0:
                cell.set_text_props(weight='bold', fontsize=table_header_fontsize, ha='left')
                cell.set_x(0.04)
            else:
                if row_headers[row-1] == "P-Value":
                    val = cell.get_text().get_text()
                    if '<' in val or (val.replace('.','',1).isdigit() and float(val) < 0.05):
                         cell.set_text_props(weight='bold', color='black')
                cell.set_text_props(fontsize=table_cell_fontsize)

    ax_table.plot([0.025, 0.975], [y_start, y_start], color='black', linewidth=2.0)
    y_hdr = y_start - ROW_HEIGHT
    ax_table.plot([0.025, 0.975], [y_hdr, y_hdr], color='black', linewidth=1.0)
    ax_table.plot([0.025, 0.975], [y_end, y_end], color='black', linewidth=2.0)

    ax_table.text(0.5, y_start + 0.05, f"{model_name} Statistical Summary", 
                  transform=ax_table.transAxes, ha='center', fontsize=title_fontsize, weight='bold')

    # 5. SAVE
    if save and OUTPUT_DIR:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(OUTPUT_DIR)
        fig_graphs.savefig(OUTPUT_DIR / f"{model_name}_Nature_Graphs.png", bbox_inches='tight', dpi=IMAGE_DPI)
        fig_table.savefig(OUTPUT_DIR / f"{model_name}_Nature_Table.png",  bbox_inches='tight', facecolor='white', dpi=IMAGE_DPI)
        print(f"✅ Saved results for {model_name}")

    if show:
        plt.show()
    else:
        plt.close(fig_graphs)
        plt.close(fig_table)




#_______________________________________________________________
# Forest Plot and Stability Proof
def plot_forest_only_refined(
    master_df: pd.DataFrame, 
    model_name: str, 
    metrics_to_plot: list, 
    save: bool = False,
    show: bool = True,
    point: bool = True,              # Attivato di default per vedere i punti
    eq_margin_multiplier: float = 0.5,
    title_fontsize: int = 22,
    axis_label_fontsize: int = 16,
    tick_fontsize: int = 14,
    legend_fontsize: int = 14,
    OUTPUT_DIR: Path = Path('Stress_Test_Results')
):
    """
    Genera il Forest Plot (Grafico B) con legenda orizzontale in basso
    e punti della distribuzione più scuri.
    """
    # [La parte di Data Preparation rimane identica alla precedente...]
    model_df = master_df[(master_df['model_name'] == model_name) & 
                         (master_df['data_split'].isin(['Test', 'Stress']))].copy()
    if model_df.empty: return

    summary_data = []
    all_drops_list = []
    
    for metric in metrics_to_plot:
        baseline = model_df[model_df['stress_level'] == 0][metric].dropna()
        stress = model_df[model_df['stress_level'] > 0][metric].dropna()
        base_seed = model_df[model_df['stress_level'] == 0][['seed', metric]].set_index('seed')
        stress_seed = model_df[model_df['stress_level'] > 0][['seed', metric]].set_index('seed')
        merged = base_seed.join(stress_seed, lsuffix='_base', rsuffix='_stress').dropna()
        drop_dist = merged[f'{metric}_base'] - merged[f'{metric}_stress']
        all_drops_list.append(pd.DataFrame({'metric': metric, 'drop_val': drop_dist.values}))
        
        n_seeds = len(drop_dist)
        sem_drop = drop_dist.std() / np.sqrt(n_seeds) if n_seeds > 0 else 0
        ci_95 = t.ppf(0.975, n_seeds-1) * sem_drop if n_seeds > 1 else 0
        _, p_val = mannwhitneyu(baseline, stress, alternative='two-sided') if not baseline.empty and not stress.empty else (None, 1.0)
        
        summary_data.append({
            'metric': metric, 'mean_drop': drop_dist.mean(), 'ci_95': ci_95,
            'eq_margin': eq_margin_multiplier * baseline.std(), 'p_value': p_val
        })
        
    summary_stats = pd.DataFrame(summary_data).set_index('metric').reindex(metrics_to_plot)
    raw_drops_df = pd.concat(all_drops_list)
    metric_labels_upper = [m.upper().replace('_', ' ') for m in metrics_to_plot]

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(11, 8)) # Allargato leggermente per la legenda orizzontale
    fig.patch.set_facecolor('white')
    
    y_pos = np.arange(len(summary_stats))
    
    # Margini di Equivalenza
    for i, margin in enumerate(summary_stats['eq_margin']):
        ax.fill_betweenx([i - 0.4, i + 0.4], -margin, margin, color='green', alpha=0.12, zorder=0)

    ax.axvline(0, color='black', linewidth=1.2, linestyle='--', zorder=1)

    # 1. PUNTI DISTRIBUZIONE (Più scuri e definiti)
    if point:
        metric_map = {m: i for i, m in enumerate(metrics_to_plot)}
        raw_drops_df['y_idx'] = raw_drops_df['metric'].map(metric_map)
        sns.stripplot(
            data=raw_drops_df, y='y_idx', x='drop_val', 
            color='#2c3e50',  # Blu-grigio scuro
            alpha=0.6,        # Più visibile
            jitter=0.2, size=5, ax=ax, zorder=2, orient='h'
        )

    # 2. MEDIA E CI
    ax.errorbar(
        summary_stats['mean_drop'], y_pos, xerr=summary_stats['ci_95'], 
        fmt='o', color='black', markersize=10, capsize=7, capthick=2, 
        elinewidth=2.5, zorder=3, label='Mean Drop (95% CI)'
    )

    # 3. LEGENDA (Spaziata e in basso)
    eq_patch = Patch(color='green', alpha=0.15, label=r'Equivalence Margin ($0.5 \times \sigma_{base}$)')
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(
        handles=handles + [eq_patch], 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.22), # Spostata più in basso
        ncol=2,                      # Distribuita su due colonne (orizzontale)
        columnspacing=3.0,           # Aumenta lo spazio tra le colonne
        frameon=False, 
        fontsize=legend_fontsize
    )

    # Raffinamento finale
    ax.set_title(f"{model_name}: Performance Drop Analysis", fontsize=title_fontsize, weight='bold', pad=25)
    ax.set_xlabel("Drop (Baseline - Stress)", fontsize=axis_label_fontsize)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_labels_upper, fontsize=tick_fontsize, fontweight='bold')
    ax.invert_yaxis() 
    ax.grid(axis='x', linestyle=':', alpha=0.4, color='grey')
    sns.despine(ax=ax, left=True) # Rimuove la linea verticale di sinistra per uno stile più moderno

    plt.tight_layout()
    
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(OUTPUT_DIR)
        fig.savefig(OUTPUT_DIR / f"{model_name}_Forest_Plot_Final.png", bbox_inches='tight', dpi=300)
    if show: plt.show()





# ======================================================================================
# Final Block
# ======================================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def master_performance_analysis(
    df: pd.DataFrame, 
    metrics: list = None, 
    ref_values: dict = None, 
    stress_level: int = 0,    
    selected_models: list = None,
    target_robustness_model: str = None, 
    
    # --- LAYOUT CONTROLS ---
    fig_height_multiplier: float = 1.0,  
    hspace: float = 0.5,                 
    wspace: float = 0.3,                 
    title_y: float = 0.98,               
    top_margin: float = 0.92,            
    section_gap_height: float = 1.0,     
    middle_header_height: float = 0.6,
    title_pad: int = 20,
    legend_gap_from_graph: float = 0.8,        
    
    # Font Sizes / Widths
    label_fontsize: int = 28,            
    title_fontsize: int = 24,
    header_fontsize: int = 26,             
    tick_fontsize: int = 14, 
    legend_fontsize: int = 14,    
    boxes_linewidth: float = 1.5,
    references_linewiths: float = 1.5,
    legend_line_width: float = 5,
    ball_size: float  = 10,
    
    # Logic Parameters
    max_per_class_diff: float = 0.2,
    robustness_ref_linewidth: float = 2.0,
    robustness_plot_width_ratio: float = 0.7, 
    stress_box_width: float = 0.7,
    
    # Output
    show: bool = False,
    save: bool = True,
    output_dir: Path = Path(".")
    ):
    
    if metrics is None: metrics = [] 
    metrics = [m.lower() if m in ['AUC', 'PR_AUC'] else m for m in metrics]
    if ref_values is None: ref_values = {}
    
    if stress_level is None: header_name = "AVERAGE (TEST + STRESS)"
    elif stress_level == 0:  header_name = "TEST SET"
    else:                    header_name = f"STRESS TEST LEVEL {stress_level}"

    df = df.copy()
    df.columns = [c.lower() if c in ['AUC', 'PR_AUC'] else c for c in df.columns]
    
    if stress_level is not None:
        analysis_df = df[df['stress_level'] == stress_level].copy()
    else:
        analysis_df = df[df['stress_level'] >= 0].copy()

    if selected_models:
        analysis_df = analysis_df[analysis_df['model_name'].isin(selected_models)]

    if 'precision_1' in analysis_df.columns:
        analysis_df['p_diff'] = (analysis_df['precision_1'] - analysis_df['precision_0']).abs()
        analysis_df['r_diff'] = (analysis_df['recall_1'] - analysis_df['recall_0']).abs()
        stats = analysis_df.groupby('model_name')[['p_diff', 'r_diff']].mean()
        bad_models = stats[(stats['p_diff'] > max_per_class_diff) | (stats['r_diff'] > max_per_class_diff)].index.tolist()
        if len(bad_models) > 0:
            analysis_df = analysis_df[~analysis_df['model_name'].isin(bad_models)]

    if analysis_df.empty: return []

    sort_metric = metrics[0].lower() if metrics else None
    order = sorted(analysis_df['model_name'].unique()) 
    if sort_metric and sort_metric in analysis_df.columns:
        if pd.api.types.is_numeric_dtype(analysis_df[sort_metric]):
            order = analysis_df.groupby('model_name')[sort_metric].mean().sort_values(ascending=False).index.tolist()

    unique_models = analysis_df['model_name'].unique()
    try:
        plot_palette = {m: MODEL_COLORS.get(m, '#333333') for m in unique_models}
        final_mapping = NAME_MAPPING_SHORT
        final_class_colors = CLASS_COLORS
    except NameError:
        plot_palette = {m: sns.color_palette("husl", len(unique_models))[i] for i, m in enumerate(unique_models)}
        final_mapping = {m: m for m in unique_models}
        final_class_colors = {'Alpha': 'blue', 'Beta': 'orange'}

    # Layout Setup
    n_cols = 3
    n_metrics = len(metrics)
    n_rows_gen = (n_metrics + n_cols - 1) // n_cols if n_metrics > 0 else 1
    row_height = 4.5 
    
    top_block_h = n_rows_gen * row_height
    leg1_height = 0.5 
    leg2_height = 1.0 
    bot_block_h = row_height + legend_gap_from_graph + leg1_height
    
    # FIXED: Increased internal multiplier gaps so elements don't crash
    outer_ratios = [
        top_block_h, 
        section_gap_height * 1.0,  # Gap 1
        leg2_height,            
        section_gap_height * 1.0,  # Gap 2
        middle_header_height, 
        section_gap_height * 1.0,  # Gap 3
        bot_block_h
    ]
    
    robustness_df = pd.DataFrame()
    if target_robustness_model:
        robustness_df = df[
            (df['model_name'] == target_robustness_model) & 
            (df['data_split'].isin(['Test', 'Stress']))
        ].copy()
        
        if not robustness_df.empty:
            stress_plot_h = row_height + 1.5 
            outer_ratios.extend([
                section_gap_height * 0.5,  # Gap before header
                middle_header_height, 
                section_gap_height * 0.8,  # FIXED: Was 0.2, causing the J title overlap!
                stress_plot_h*0.9
            ])
    
    total_height = sum(outer_ratios) * fig_height_multiplier
    fig = plt.figure(figsize=(22, total_height))
    
    # SupTitle logic
    fig.suptitle(f"{header_name} PERFORMANCE OVERVIEW", fontsize=header_fontsize, weight='bold', y=title_y)

    gs_outer = fig.add_gridspec(len(outer_ratios), 1, height_ratios=outer_ratios, hspace=0.0, top=top_margin, bottom=0.05, left=0.05, right=0.95)
    plot_idx = 0 
    
    box_props = dict(facecolor='none', edgecolor='black', linewidth=boxes_linewidth)
    whisker_props = dict(linewidth=boxes_linewidth, color='black')
    median_props = dict(linewidth=boxes_linewidth, color='black')

    # BLOCK 1: TOP GRAPHS
    if metrics:
        gs_top = gridspec.GridSpecFromSubplotSpec(n_rows_gen, n_cols, subplot_spec=gs_outer[0], hspace=hspace, wspace=wspace)
        for i, metric in enumerate(metrics):
            if metric not in analysis_df.columns: continue
            r, c = i // n_cols, i % n_cols
            ax = fig.add_subplot(gs_top[r, c])
            
            ax.set_title(f"{chr(65 + plot_idx)}.", loc='left', fontsize=label_fontsize, weight='bold', pad=title_pad)
            ax.set_title(metric.upper().replace('_', ' '), loc='center', fontsize=title_fontsize, weight='bold', pad=title_pad)
            plot_idx += 1
            
            sns.stripplot(data=analysis_df, x='model_name', y=metric, order=order, hue='model_name', palette=plot_palette, size=ball_size, alpha=1.0, jitter=0.25, ax=ax, legend=False, zorder=0)
            sns.boxplot(data=analysis_df, x='model_name', y=metric, order=order, width=0.5, showfliers=False, ax=ax, zorder=10, boxprops=box_props, whiskerprops=whisker_props, medianprops=median_props)
            
            if metric in ref_values:
                ref = (ref_values[metric].get('alpha', 0) + ref_values[metric].get('beta', 0)) / 2 if isinstance(ref_values[metric], dict) else ref_values[metric]
                if ref > 0: ax.axhline(ref, color='red', linestyle='--', linewidth=references_linewiths, alpha=0.7)

            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([final_mapping.get(m, m) for m in order], rotation=30, ha='right', fontsize=tick_fontsize)
            ax.tick_params(axis='both', which='major', width=boxes_linewidth, length=6, labelsize=tick_fontsize, bottom=True, left=True)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(boxes_linewidth); ax.spines['bottom'].set_linewidth(boxes_linewidth)

    # BLOCK 2: MODEL LEGEND
    ax_leg_models = fig.add_subplot(gs_outer[2])
    ax_leg_models.axis('off')
    model_handles = [mlines.Line2D([], [], color=plot_palette.get(m, '#333'), marker='o', linestyle='None', markersize=16, label=f"{final_mapping.get(m, m)} : {m}") for m in order]
    ax_leg_models.legend(handles=model_handles, loc='center', ncol=min(len(order), 5), frameon=False, fontsize=legend_fontsize)

    # BLOCK 3 & 4: PER-CLASS
    ax_sp = fig.add_subplot(gs_outer[4])
    ax_sp.axis('off')
    ax_sp.text(0.5, 0.5, "PER-CLASS PERFORMANCE BREAKDOWN", ha='center', va='center', fontsize=header_fontsize, weight='bold')

    bot_ratios = [row_height, legend_gap_from_graph, leg1_height]
    gs_bot = gridspec.GridSpecFromSubplotSpec(3, n_cols, subplot_spec=gs_outer[6], height_ratios=bot_ratios, hspace=0.0, wspace=wspace)
    
    pc_metrics = ['precision', 'recall', 'f1']
    pc_palette = {'Alpha': final_class_colors.get('Alpha', 'blue'), 'Beta': final_class_colors.get('Beta', 'orange')}

    for i, m_base in enumerate(pc_metrics):
        if i >= n_cols: break
        ax = fig.add_subplot(gs_bot[0, i])
        
        ax.set_title(f"{chr(65 + plot_idx)}.", loc='left', fontsize=label_fontsize, weight='bold', pad=title_pad)
        ax.set_title(f"Per-Class {m_base.capitalize()}", loc='center', fontsize=title_fontsize, weight='bold', pad=title_pad)
        plot_idx += 1
        
        col0, col1 = f"{m_base}_0", f"{m_base}_1"
        if col0 in analysis_df.columns:
            melted = analysis_df.melt(id_vars=['model_name'], value_vars=[col0, col1], var_name='Class_Type', value_name='Score')
            melted['Class'] = melted['Class_Type'].map({col0: 'Alpha', col1: 'Beta'})
            sns.boxplot(data=melted, x='model_name', y='Score', hue='Class', order=order, palette=pc_palette, ax=ax, linewidth=boxes_linewidth, showfliers=False)
        
        if m_base in ref_values and isinstance(ref_values[m_base], dict):
            if 'alpha' in ref_values[m_base]: ax.axhline(ref_values[m_base]['alpha'], color=final_class_colors['Alpha'], ls=':', lw=references_linewiths)
            if 'beta' in ref_values[m_base]:  ax.axhline(ref_values[m_base]['beta'], color=final_class_colors['Beta'], ls='--', lw=references_linewiths)

        ax.set_xlabel(""); ax.set_ylabel("") 
        if ax.get_legend(): ax.get_legend().remove()
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([final_mapping.get(m, m) for m in order], rotation=30, ha='right', fontsize=tick_fontsize)
        ax.tick_params(axis='both', which='major', width=boxes_linewidth, length=6, labelsize=tick_fontsize, bottom=True, left=True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(boxes_linewidth); ax.spines['bottom'].set_linewidth(boxes_linewidth)

    ax_leg1 = fig.add_subplot(gs_bot[2, :])
    ax_leg1.axis('off')
    h_alpha = mlines.Line2D([], [], color=final_class_colors['Alpha'], ls=':', lw=legend_line_width, label='Alpha Class')
    h_beta  = mlines.Line2D([], [], color=final_class_colors['Beta'], ls='--', lw=legend_line_width, label='Beta Class')
    h_ref   = mlines.Line2D([], [], color='red', ls='--', lw=legend_line_width, label='Ref. Benchmark')
    ax_leg1.legend(handles=[h_alpha, h_beta, h_ref], loc='center', ncol=3, frameon=False, fontsize=legend_fontsize)

    # BLOCK 5: STRESS ROBUSTNESS
    if not robustness_df.empty:
        ax_stress_hdr = fig.add_subplot(gs_outer[8])
        ax_stress_hdr.axis('off')
        ax_stress_hdr.text(0.5, 0.5, f"{target_robustness_model} Stress Robustness", ha='center', va='center', fontsize=header_fontsize, weight='bold')

        w = robustness_plot_width_ratio
        spacer = (1 - w) / 2
        gs_stress_inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_outer[10], 
            width_ratios=[spacer, w, spacer]
        )
        ax_stress = fig.add_subplot(gs_stress_inner[0, 1])
        
        ax_stress.set_title(f"{chr(65 + plot_idx)}.", loc='left', fontsize=label_fontsize, weight='bold', pad=title_pad)
        melted_stress = pd.melt(robustness_df, id_vars=['seed', 'stress_level'], value_vars=metrics, var_name='metric', value_name='score')
        metric_labels_upper = [m.upper().replace('_', ' ') for m in metrics]

        n_levels = robustness_df['stress_level'].nunique()
        stress_colors = sns.color_palette("Blues", n_colors=n_levels + 2)[2:]
        unique_levels = sorted(robustness_df['stress_level'].unique())
        custom_palette = {lvl: ('#d9d9d9' if lvl == 0 else stress_colors[i-1]) for i, lvl in enumerate(unique_levels)}
        
        sns.boxplot(
            data=melted_stress, x='metric', y='score', hue='stress_level', width=stress_box_width,
            order=metrics, palette=custom_palette, ax=ax_stress, fliersize=3, linewidth=boxes_linewidth,
            boxprops=dict(edgecolor='black', alpha=0.9, linewidth=boxes_linewidth), 
            medianprops=dict(color='black', linewidth=boxes_linewidth * 1.5),
            whiskerprops=dict(color='black', linewidth=boxes_linewidth), capprops=dict(color='black', linewidth=boxes_linewidth)
        )

        ax_stress.grid(False) 
        sns.despine(ax=ax_stress, top=True, right=True, left=False, bottom=False)
        ax_stress.spines['left'].set_linewidth(boxes_linewidth)
        ax_stress.spines['bottom'].set_linewidth(boxes_linewidth)

        if ref_values:
            for i, metric in enumerate(metrics):
                if metric in ref_values:
                    ref_val = np.mean(list(ref_values[metric].values())) if isinstance(ref_values[metric], dict) else ref_values[metric]
                    ax_stress.hlines(y=ref_val, xmin=i - 0.4, xmax=i + 0.4, color='red', linestyle='--', linewidth=robustness_ref_linewidth, zorder=10)

        ax_stress.set_xlabel(""); ax_stress.set_ylabel("Absolute Score", fontsize=tick_fontsize)
        ax_stress.set_xticklabels(metric_labels_upper, fontsize=tick_fontsize, fontweight='bold', rotation=0, ha='center')
        ax_stress.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=boxes_linewidth, length=6)
        
        ax_stress.legend(
            title='Stress Level', 
            loc='upper center',          # The point on the legend we are anchoring
            bbox_to_anchor=(0.5, -0.2),  # Places it below the X-axis (0.5 is center)
            ncol=n_levels,               # Spreads items out horizontally
            frameon=False, 
            fontsize=legend_fontsize, 
            title_fontsize=legend_fontsize
        )
    if save and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"Master_Performance_{header_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        path = output_dir / fname
        fig.savefig(path, bbox_inches='tight', dpi=300) 
        print(f"✅ Saved Master Graph: {path}")

    if show: plt.show()
    plt.close(fig)
    return order


