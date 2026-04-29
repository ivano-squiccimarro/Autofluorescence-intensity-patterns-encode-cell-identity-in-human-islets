# ___________________________________________________________________________
# Imports

import joblib
import numpy    as np
import pandas   as pd
import seaborn  as sns

import matplotlib.pyplot    as plt
import matplotlib.patches   as mpatches
import matplotlib.gridspec  as gridspec

from   matplotlib.lines     import Line2D
from   matplotlib.patches   import Ellipse

from sklearn.decomposition          import PCA
from sklearn.cluster                import KMeans
from sklearn.preprocessing          import StandardScaler
from sklearn.mixture                import GaussianMixture
from sklearn.metrics.pairwise       import pairwise_distances
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis as LDA

from scipy.stats import chi2_contingency, mannwhitneyu

from src.feature_extraction_functions import extract_feats
from config.presentation_config       import (
    set_nature_style,
    COLOR_BLIND_PALETTE,
    CLASS_COLORS,
    publication_names_master,
    TECH_TO_PUB, 
    technical_names_master,
    IMAGE_DPI)

# ___________________________________________________________________________
# 1. DATA LOADING & EXTRACTION FUNCTION
def get_feature_dataframe(image_df_path, pub_features_to_keep):
    """
    Loads raw images, RUNS EXTRACTION, renames features to publication names, 
    and filters based on the consistency check.
    """
    if not image_df_path.exists():
        print(f"❌ ERROR: Could not find {image_df_path}")
        return None
        
    print(f"\nLoading raw images from {image_df_path}...")
    full_original_df = joblib.load(image_df_path)
    
    print(f"⚙️ Running Feature Extraction on {len(full_original_df)} images...")
    # This generates the technical feature columns (e.g., shp_area, ltp_u_0...)
    features_only_df = extract_feats(full_original_df, n_jobs=-1, verbosity=0)
    
    # 1. Separate Label
    if 'label' in features_only_df.columns:
        y = features_only_df['label']
        X_raw = features_only_df.drop('label', axis=1)
    else:
        print("❌ Error: 'label' column missing from extracted dataframe.")
        return None

    # 2. Safety Check: Column Count
    # We compare against the global master list defined in S2
    if X_raw.shape[1] != len(publication_names_master):
        print(f"⚠️ Warning: Feature count mismatch.")
        print(f"   Expected: {len(publication_names_master)}")
        print(f"   Got:      {X_raw.shape[1]}")

    # 3. Robust Renaming Logic
    # We need to map whatever extract_feats produced -> Publication Names
    first_col = X_raw.columns[0]
    rename_dict = {}

    if isinstance(first_col, int):
        print("   ℹ️ Detected Integer columns. Mapping by position.")
        rename_dict = {i: name for i, name in enumerate(publication_names_master)}
    else:
        print("   ℹ️ Detected Technical strings. Mapping via dictionary.")
        # Ensure we use the global map defined in S2
        if 'TECH_TO_PUB' in globals():
            rename_dict = TECH_TO_PUB
        else:
            # Fallback reconstruction if S2 wasn't run
            rename_dict = dict(zip(technical_names_master, publication_names_master))

    # Apply Rename
    X_raw.rename(columns=rename_dict, inplace=True)

    # 4. Filter Columns
    available_features = set(X_raw.columns)
    valid_kept = [f for f in pub_features_to_keep if f in available_features]
    
    # Validation
    if not valid_kept:
        print("❌ Error: No matching features found to keep!")
        print(f"   Available (First 5): {list(available_features)[:5]}")
        print(f"   Requested (First 5): {pub_features_to_keep[:5]}")
        return None
    
    missing = set(pub_features_to_keep) - set(valid_kept)
    if missing:
        print(f"⚠️ Warning: {len(missing)} features were requested but not found.")

    final_df = X_raw[valid_kept].copy()
    final_df['label'] = y
    
    print(f"✅ DataFrame constructed. Shape: {final_df.shape}")
    return final_df

def label_distribution(features_df):
    print("\n" + "="*40)
    print("S3.1: Dataset Composition")
    print("="*40)
    
    counts = features_df['label'].value_counts().sort_index()
    alpha_count = counts.get(0, 0)
    beta_count = counts.get(1, 0)
    total = alpha_count + beta_count
    
    if total == 0: return

    print("\n| Class | Cell Type | N | % |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| 0 | Alpha | {alpha_count} | {alpha_count/total:.1%} |")
    print(f"| 1 | Beta  | {beta_count} | {beta_count/total:.1%} |")
    print(f"| **Total** | | **{total}** | **100%** |")


# ___________________________________________________________________________
# 3. PCA VISUALIZATION

def PCA_features_visualization(
    features_df, 
    output_dir, 
    text_size=14, 
    title_fontsize=20, 
    x_tick_fontsize=12,
    y_tick_fontsize=12,
    pad= 10,
    show=True, 
    save=True
):
    set_nature_style()

    print("\nS3.4: Dimensionality Reduction (PCA)")
    
    # 1. Data Prep
    X = features_df.drop(['label'], axis=1, errors='ignore')
    y = features_df['label'].astype(int)
    X_scaled = StandardScaler().fit_transform(X)
    
    c_alpha = CLASS_COLORS['Alpha']
    c_beta  = CLASS_COLORS['Beta']

    fig = plt.figure(figsize=(18, 8))
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.25)

    # ==========================================================================
    # PANEL A: SCREE PLOT
    # ==========================================================================
    ax1 = fig.add_subplot(gs_main[0])
    ax1.text(-0.1, 1.05, 'A.         Explained Variance by Principal Component', 
             transform=ax1.transAxes, fontsize=title_fontsize, fontweight='bold')
    
    pca_full = PCA(n_components=None, random_state=42).fit(X_scaled)
    var_exp = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(var_exp)
    
    n_95 = np.searchsorted(cum_var, 0.95) + 1
    plot_n = min(n_95 + 5, 20)
    
    ax1.bar(range(1, plot_n+1), var_exp[:plot_n], 
            color='skyblue', edgecolor='black', 
            linewidth=0.5, label='Individual')
    ax1.plot(range(1, plot_n+1), cum_var[:plot_n], 
             marker='o', color='black', 
             markersize=5, label='Cumulative')
    
    ax1.set_xlabel("Principal Components", fontsize=text_size, labelpad= pad)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=text_size, labelpad= pad)
    ax1.set_ylim(0, 1.05)
    
    # --- TICK FIXES FOR PANEL A ---
    # 1. Force Ticks Visibility
    ax1.tick_params(axis='x', labelsize=x_tick_fontsize, direction='out', width=1.2, length=6, bottom=True)
    ax1.tick_params(axis='y', labelsize=y_tick_fontsize, direction='out', width=1.2, length=6, left=True)
    
    # 2. Custom Integer Ticks (1, 2, 3, 4, 5, 10, 15, 20...)
    # We generate a list containing 1-5, then steps of 5
    ticks_to_show = [i for i in range(1, 6) if i <= plot_n] + [i for i in range(10, plot_n + 1, 5)]
    ax1.set_xticks(ticks_to_show)
    
    # Ensure Spines are visible and thick enough
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_visible(True)
        ax1.spines[spine].set_linewidth(1.0)
        ax1.spines[spine].set_color('black')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax1.legend(frameon=False, fontsize=text_size-2)

    # ==========================================================================
    # PANEL B: 2D EMBEDDING
    # ==========================================================================
    gs_joint = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[1], 
                                                width_ratios=[6, 1], height_ratios=[1, 6],
                                                wspace=0.05, hspace=0.05)
    
    ax_scatter = fig.add_subplot(gs_joint[1, 0])
    ax_hist_x  = fig.add_subplot(gs_joint[0, 0], sharex=ax_scatter)
    ax_hist_y  = fig.add_subplot(gs_joint[1, 1], sharey=ax_scatter)
    
    ax_hist_x.text(-0.1, 1.36, 'B.        2D-PCA Embedding by Cell Type', 
                   transform=ax_hist_x.transAxes, fontsize=title_fontsize, fontweight='bold')
    
    pca_2d = PCA(n_components=2, random_state=42)
    embedding = pca_2d.fit_transform(X_scaled)
    plot_df = pd.DataFrame(embedding, columns=['PC1', 'PC2'])
    plot_df['Label'] = y.map({0: 'Alpha', 1: 'Beta'}).values
    
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Label', 
                    palette={'Alpha': c_alpha, 'Beta': c_beta},
                    s=30, alpha=0.6, edgecolor='black', linewidth=0.3,
                    ax=ax_scatter, legend=True)
    
    sns.kdeplot(data=plot_df, x='PC1', hue='Label', palette={'Alpha': c_alpha, 'Beta': c_beta},
                fill=True, alpha=0.3, linewidth=0, ax=ax_hist_x, legend=False)
    sns.kdeplot(data=plot_df, y='PC2', hue='Label', palette={'Alpha': c_alpha, 'Beta': c_beta},
                fill=True, alpha=0.3, linewidth=0, ax=ax_hist_y, legend=False)
    
    ax_hist_x.axis('off'); ax_hist_y.axis('off')
    
    # --- FOCUS LOGIC (Zooming in) ---
    # Calculate bounds with a small 10% margin to remove excess whitespace
    x_min, x_max = plot_df['PC1'].min(), plot_df['PC1'].max()
    y_min, y_max = plot_df['PC2'].min(), plot_df['PC2'].max()
    margin = 0.1
    x_pad = (x_max - x_min) * margin
    y_pad = (y_max - y_min) * margin
    
    ax_scatter.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_scatter.set_ylim(y_min - y_pad, y_max + y_pad)

    # Labeling
    ax_scatter.set_xlabel(f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.1%})", 
                          fontsize=text_size, labelpad = pad)
    ax_scatter.set_ylabel(f"Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.1%})", 
                          fontsize=text_size, labelpad = pad)
    
    # --- TICK & AXIS CONTROL (Panel B) ---
    ax_scatter.tick_params(axis='x', labelsize=x_tick_fontsize, direction='out', width=1.2, length=6, bottom=True)
    ax_scatter.tick_params(axis='y', labelsize=y_tick_fontsize, direction='out', width=1.2, length=6, left=True)
    
    for spine in ['bottom', 'left']:
        ax_scatter.spines[spine].set_visible(True)
        ax_scatter.spines[spine].set_linewidth(1.0)
        ax_scatter.spines[spine].set_color('black')
        
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    
    ax_scatter.legend(frameon=False, loc='upper right', fontsize=text_size)

    if save: plt.savefig(output_dir / "PCA_Nature.png", bbox_inches='tight', dpi=IMAGE_DPI)
    if show: plt.show()
    plt.close(fig)

# ___________________________________________________________________________
# 3. PCA VISUALIZATION

def linear_separability_test_publication(
    features_df, 
    output_dir, 
    # --- FONT SIZE CONTROLS ---
    title_fontsize=20,       
    axis_label_fontsize=18,  # For X/Y Axis Labels
    tick_fontsize=18,        # For Axis Ticks (numbers)
    legend_fontsize=16,      # For the Plot Legend
    table_fontsize=20,       # For the Table Text
    show=True, 
    save=True
):
    
    set_nature_style()
    print("\nS3.5: Linear Separability Analysis")
    
    # --- Data Prep ---
    X = features_df.drop(['label'], axis=1, errors='ignore')
    y = features_df['label']
    X_scaled = StandardScaler().fit_transform(X)
    
    lda = LDA(n_components=1)
    proj = lda.fit_transform(X_scaled, y).flatten()
    
    m0, m1 = np.mean(proj[y==0]), np.mean(proj[y==1])
    v0, v1 = np.var(proj[y==0]), np.var(proj[y==1])
    fisher = np.abs(m0 - m1)**2 / (v0 + v1)
    
    X_sub, y_sub = (X_scaled, y.values) if len(X_scaled) < 2000 else (X_scaled[:2000], y.values[:2000])
    d_inter = pairwise_distances(X_sub[y_sub==0], X_sub[y_sub==1]).mean()
    d_intra = (pairwise_distances(X_sub[y_sub==0]).mean() + pairwise_distances(X_sub[y_sub==1]).mean()) / 2
    
    # --- Figure Setup ---
    fig = plt.figure(figsize=(18, 7)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.15)
    
    ax_plot = plt.subplot(gs[0])
    ax_table = plt.subplot(gs[1])
    
    # ==========================================================================
    # PANEL A: LDA DENSITY PLOT
    # ==========================================================================
    ax_plot.text(0.1, 1.08, 'A. Linear Discriminant Analysis Projection', 
                 transform=ax_plot.transAxes, fontsize=title_fontsize, fontweight='bold')
    
    plot_df = pd.DataFrame({'Score': proj, 'Label': y.map({0: 'Alpha', 1: 'Beta'}).values})
    
    sns.kdeplot(data=plot_df, x='Score', hue='Label', fill=True,
                palette={'Alpha': CLASS_COLORS['Alpha'], 'Beta': CLASS_COLORS['Beta']},
                alpha=0.5, linewidth=1.5, ax=ax_plot, legend=False)
    
    ax_plot.axvline((m0+m1)/2, color='black', linestyle='--', linewidth=1.5)
    
    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS['Alpha'], label='Alpha', alpha=0.5),
               mpatches.Patch(color=CLASS_COLORS['Beta'], label='Beta', alpha=0.5)]
    ax_plot.legend(handles=handles, loc='upper right', frameon=False, fontsize=legend_fontsize)
    
    # Labels
    ax_plot.set_xlabel("LDA Discriminant Score", fontsize=axis_label_fontsize, labelpad=10)
    ax_plot.set_ylabel("Density", fontsize=axis_label_fontsize, labelpad=10)
    
    # --- AXES & TICKS CLEANUP ---
    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    ax_plot.spines['left'].set_visible(True)   
    ax_plot.spines['bottom'].set_visible(True) 
    
    # Apply Tick Font Size
    ax_plot.tick_params(axis='y', left=True, labelleft=True, labelsize=tick_fontsize, direction='out')
    ax_plot.tick_params(axis='x', bottom=True, labelbottom=True, labelsize=tick_fontsize, direction='out')

    # ==========================================================================
    # PANEL B: METRICS TABLE
    # ==========================================================================
    ax_table.axis('off')
    ax_table.text(0.1, 1.08, 'B. Quantitative Separability Metrics', 
                  transform=ax_table.transAxes, fontsize=title_fontsize, fontweight='bold')
    
    col_labels = ["Metric", "Value"]
    cell_text = [
        ["Fisher's Ratio", f"{fisher:.3f}"],
        ["Separation Ratio (Inter/Intra)", f"{d_inter/d_intra:.3f}"],
        ["Mean Intra-Class Dist.", f"{d_intra:.3f}"],
        ["Mean Inter-Class Dist.", f"{d_inter:.3f}"]
    ]
    
    # --- TABLE LAYOUT ---
    col_widths = [0.75, 0.25] 
    bbox = [0.05, 0.3, 0.9, 0.5] 
    
    table = ax_table.table(
        cellText=cell_text, 
        colLabels=col_labels, 
        loc='center', 
        cellLoc='center', 
        colWidths=col_widths,
        bbox=bbox
    )
    
    # Apply Table Font Size
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)
    
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('white') # Transparent borders
        if key[0] == 0: 
            cell.set_text_props(weight='bold')
            
    # --- DRAWING LINES (CALCULATED) ---
    x_left   = bbox[0]
    x_right  = bbox[0] + bbox[2]
    y_top    = bbox[1] + bbox[3]      
    y_bottom = bbox[1]               
    
    row_height = bbox[3] / 5 # 5 rows (1 header + 4 data)
    y_header = y_top - row_height
    
    # Plot Lines using transAxes
    kwargs = dict(transform=ax_table.transAxes, color='black')
    ax_table.plot([x_left, x_right], [y_top, y_top], linewidth=2.0, **kwargs)       # Top
    ax_table.plot([x_left, x_right], [y_header, y_header], linewidth=1.0, **kwargs) # Header
    ax_table.plot([x_left, x_right], [y_bottom, y_bottom], linewidth=2.0, **kwargs) # Bottom

    if save: 
        plt.savefig(output_dir / "Linear_Separability_Nature.png", bbox_inches='tight', dpi = IMAGE_DPI)
    if show: 
        plt.show()
    plt.close(fig)

# ___________________________________________________________________________
# 3. Kmeans and GMM Unsupervised CLustering Functions

def unsupervised_clustering_analysis_one_row(
    features_df, 
    output_dir, 
    k_means=True, 
    gmm=True,
    # --- FONT SIZE CONTROLS ---
    super_title_fontsize=24, 
    title_fontsize=20,       
    label_fontsize=16,       
    tick_fontsize=14,         
    legend_fontsize=14,       
    pad=10,
    y_supertitle = 1.02,
    show=True, 
    save=True
):
    
    set_nature_style()
    print("\nS3.7: Clustering Analysis")
    
    # --- Data Prep ---
    X = features_df.drop(['label'], axis=1, errors='ignore')
    y_true = features_df['label'].values if 'label' in features_df.columns else None
    X_scaled = StandardScaler().fit_transform(X)
    pca_emb = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    
    # Palette for clusters
    cluster_pal = sns.color_palette("deep", 10) 

    # ======================================================================
    # PART 1: K-MEANS ANALYSIS
    # ======================================================================
    if k_means:
        fig = plt.figure(figsize=(24, 8), layout='constrained') 
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.8, 1.2], figure=fig)
        fig.suptitle("S3.7: Unsupervised Clustering Analysis (K-Means)", 
                     fontsize=super_title_fontsize, fontweight='bold', y=y_supertitle)
        
        # --- PLOT A: ELBOW METHOD ---
        ax1 = fig.add_subplot(gs[0])
        ax1.text(0.25, 1.05, 'A. Elbow Method', transform=ax1.transAxes, 
                 fontsize=title_fontsize, fontweight='bold')
        
        k_range = range(1, 16)
        wcss = [KMeans(n_clusters=k, n_init=3, random_state=42).fit(X_scaled).inertia_ for k in k_range]
        
        # Kneedle Detection
        p1, p2 = np.array([k_range[0], wcss[0]]), np.array([k_range[-1], wcss[-1]])
        line_vec = p2 - p1
        distances = [np.abs(np.cross(line_vec, p1 - np.array([k, wcss[i]]))) / np.linalg.norm(line_vec) 
                     for i, k in enumerate(k_range)]
        k_best = k_range[np.argmax(distances)]
        
        ax1.plot(k_range, wcss, '-o', color='black', lw=1.5, markersize=6, label='Inertia')
        ax1.plot(k_best, wcss[k_best-1], 'o', color=COLOR_BLIND_PALETTE.get('amber'), markersize=18, 
                 markeredgecolor='black', markeredgewidth=1.5, label=f'Optimal k={k_best}')
        
        ax1.set_xlabel("Number of Clusters (k)", fontsize=label_fontsize, labelpad=pad)
        ax1.set_ylabel("Inertia (WCSS)", fontsize=label_fontsize, labelpad=pad)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.legend(frameon=False, fontsize=legend_fontsize)

        # --- PLOT B: IMPURITY ---
        labels_km = KMeans(n_clusters=k_best, n_init=10, random_state=42).fit_predict(X_scaled)
        scores = []
        for i in range(k_best):
            sub = y_true[labels_km == i] if y_true is not None else []
            scores.append(1 - ((sub==0).mean()**2 + (sub==1).mean()**2) if len(sub)>0 else 0)
            
        ax2 = fig.add_subplot(gs[1])
        ax2.text(0.25, 1.05, f'B. Cluster Impurity (k={k_best})', transform=ax2.transAxes, 
                 fontsize=title_fontsize, fontweight='bold')
        ax2.bar(range(1, k_best+1), scores, width=0.4, color='skyblue', edgecolor='black', lw=0.5)
        ax2.axhline(np.mean(scores), color='black', ls='--', lw=2, label='Mean Impurity')
        ax2.set_ylim(0, 0.55)
        ax2.set_xlabel("Cluster ID", fontsize=label_fontsize)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.legend(frameon=False, fontsize=legend_fontsize)

        # --- PLOT C: CLUSTER MAP ---
        ax3 = fig.add_subplot(gs[2])
        ax3.text(0.35, 1.05, 'C. K-Means Cluster Map', transform=ax3.transAxes, 
                 fontsize=title_fontsize, fontweight='bold')
        current_pal = cluster_pal[:k_best] if k_best <= 10 else sns.color_palette("husl", k_best)
        sns.scatterplot(x=pca_emb[:,0], y=pca_emb[:,1], hue=labels_km, palette=current_pal,
                        s=80, alpha=0.7, edgecolor='white', lw=0.5, ax=ax3)
        ax3.set_xlabel("PC1", fontsize=label_fontsize)
        ax3.set_ylabel("PC2", fontsize=label_fontsize)
        ax3.tick_params(axis='both', labelsize=tick_fontsize)
        ax3.spines[['top', 'right']].set_visible(False)
        leg = ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, title="Cluster ID", 
                   fontsize=legend_fontsize)
        leg.get_title().set_fontsize('22')

        if save: plt.savefig(output_dir / "Clustering_KMeans_Nature.png", bbox_inches='tight', dpi=IMAGE_DPI)
        if show: plt.show()
        plt.close(fig)

    # ======================================================================
    # PART 2: GMM ANALYSIS
    # ======================================================================
    if gmm:
        fig = plt.figure(figsize=(24, 8), layout='constrained') 
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.8, 1.2], figure=fig)
        fig.suptitle("S3.7: Unsupervised Clustering Analysis (GMM)", 
                     fontsize=super_title_fontsize, fontweight='bold', y=y_supertitle)
        
        # --- PLOT A: BIC SCORE ---
        ax1 = fig.add_subplot(gs[0])
        ax1.text(0.25, 1.05, 'A. Model Selection (BIC)', transform=ax1.transAxes, 
                 fontsize=title_fontsize, fontweight='bold')
        
        n_range = range(1, 16)
        bic_scores = [GaussianMixture(n_components=n, n_init=3, random_state=42).fit(X_scaled).bic(X_scaled) for n in n_range]
        n_best = n_range[np.argmin(bic_scores)]
        
        ax1.plot(n_range, bic_scores, '-o', color='black', lw=1.5, markersize=6, label='BIC Score')
        ax1.plot(n_best, bic_scores[n_best-1], 'o', color=COLOR_BLIND_PALETTE.get('amber'), markersize=18, 
                 markeredgecolor='black', markeredgewidth=1.5, label=f'Optimal n={n_best}')
        
        ax1.set_xlabel("Number of Components (n)", fontsize=label_fontsize, labelpad=pad)
        ax1.set_ylabel("BIC Value", fontsize=label_fontsize, labelpad=pad)
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.legend(frameon=False, fontsize=legend_fontsize)

        # --- PLOT B: IMPURITY ---
        labels_gmm = GaussianMixture(n_components=n_best, n_init=5, random_state=42).fit_predict(X_scaled)
        scores_g = []
        for i in range(n_best):
            sub = y_true[labels_gmm == i] if y_true is not None else []
            scores_g.append(1 - ((sub==0).mean()**2 + (sub==1).mean()**2) if len(sub)>0 else 0)

        ax2 = fig.add_subplot(gs[1])
        ax2.text(0.25, 1.1, f'B. Cluster Impurity (n={n_best})', transform=ax2.transAxes, 
                 fontsize=title_fontsize, fontweight='bold')
        
        ax2.bar(range(1, n_best+1), scores_g, width=0.4, color='skyblue', edgecolor='black', lw=0.5)
        ax2.axhline(np.mean(scores_g), color=COLOR_BLIND_PALETTE.get('red', 'red'), ls='--', lw=2, label='Mean Impurity')
        ax2.set_ylim(0, 0.55)
        ax2.set_xlabel("Cluster ID", fontsize=label_fontsize)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.legend(frameon=False, fontsize=legend_fontsize)

        # --- PLOT C: CLUSTER MAP ---
        ax3 = fig.add_subplot(gs[2])
        ax3.text(0.35, 1.05, 'C. GMM Cluster Map', transform=ax3.transAxes, 
                 fontsize=title_fontsize, fontweight='bold')
        current_pal_g = cluster_pal[:n_best] if n_best <= 10 else sns.color_palette("husl", n_best)
        sns.scatterplot(x=pca_emb[:,0], y=pca_emb[:,1], hue=labels_gmm, palette=current_pal_g,
                        s=80, alpha=0.7, edgecolor='white', lw=0.5, ax=ax3)
        ax3.set_xlabel("PC1", fontsize=label_fontsize)
        ax3.tick_params(axis='both', labelsize=tick_fontsize)
        ax3.spines[['top', 'right']].set_visible(False)
        leg = ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, 
                         title="Cluster ID", fontsize=legend_fontsize)
        leg.get_title().set_fontsize('22')

        if save: plt.savefig(output_dir / "Clustering_GMM_Nature.png", bbox_inches='tight', dpi=IMAGE_DPI)
        if show: plt.show()
        plt.close(fig)


# All font sizes in one place — adjust here to restyle globally
FS = dict(
    suptitle = 13,
    title    = 12,
    label    = 11,
    tick     = 10,
    annot    =  9,
    panel    = 14,  
)

plt.rcParams.update({
    "font.family"       : "sans-serif",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 0.8,
    "xtick.major.width" : 0.8,
    "ytick.major.width" : 0.8,
})

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1  FIT GMM ONCE — all downstream functions receive these objects
# ──────────────────────────────────────────────────────────────────────────────
def fit_gmm(
    features_df: pd.DataFrame,
    n_clusters:  int = 3,
    random_state: int = 42,
):
    """
    Scale → GMM → PCA.  Clusters are re-labelled so that 0 = Alpha-enriched
    and 2 = Beta-enriched (ascending mean PC1 score).

    Returns
    -------
    cluster_labels : np.ndarray   (n_samples,)  — 0 / 1 / 2
    X_scaled       : np.ndarray   (n_samples, n_features)
    pca            : fitted PCA(n_components=2)
    pc1_scores     : np.ndarray   (n_samples,)
    """
    X = features_df.drop(columns=["label"], errors="ignore")
    X_scaled = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(n_components=n_clusters, n_init=10,
                          random_state=random_state)
    raw = gmm.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(X_scaled)
    pc1 = pca.transform(X_scaled)[:, 0]

    order  = (pd.DataFrame({"c": raw, "pc1": pc1})
              .groupby("c")["pc1"].mean()
              .sort_values().index.tolist())
    remap  = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[c] for c in raw])

    return labels, X_scaled, pca, pc1

# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _confidence_ellipse(x, y, ax, n_std=2.0, **kw):
    """Covariance-correct confidence ellipse (eigendecomposition)."""
    cov  = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    angle  = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    w, h   = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), w, h, angle=angle, **kw))

C_CLUST_GLOBAL = ["#88CCEE", "#DDCC77", "#CC6677"]
L_CLUST_GLOBAL = ["Alpha-enriched", "Intermediate", "Beta-enriched"]


def make_unified_summary_figure(
    features_df:    pd.DataFrame,
    cluster_labels: np.ndarray,
    X_scaled:       np.ndarray,
    pca,
    output_dir      = None,
    pca_width_ratio: float = 1.5,
    title_fontsize:  int = 18,
    label_fontsize:  int = 14,
    tick_fontsize:   int = 12,
    annot_fontsize:  int = 14,
    legend_fontsize: int = 13,
    table_fontsize:  int = 12,
    show:           bool = True,
    save:           bool = True,
):
    """
    Creates a master publication figure following Nature Journal standards:
      - Times New Roman, serif fonts.
      - Top Left: 2D-PCA
      - Top Right: Stacked Barplot for GMM Enrichment
      - Middle Left: Legend
      - Middle Right: Clean Chi-Square Stats (no box)
      - Bottom: Academic three-line table
    """
    print("\n" + "=" * 50)
    print("Generating Master Figure: PCA, Enrichment & APA Table")
    print("=" * 50)

    # ── 1. Apply Nature Journal Style ──────────────────────────────────────
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set_context("paper")
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': tick_fontsize, # Base font size
        'pdf.fonttype': 42,      # Ensures fonts are embedded in PDFs
        'ps.fonttype': 42
    })

    # Fallback palettes
    C_CLASS = {"Alpha": "#E69F00", "Beta": "#0072B2"} 
    C_CLUST = ["#88CCEE", "#DDCC77", "#CC6677"]

    # ── Pre-compute shared quantities ──────────────────────────────────────
    X_pca = pca.transform(X_scaled)
    y_map = features_df["label"].map({0: "Alpha", 1: "Beta"})

    df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": cluster_labels, "Label": y_map.values})
    
    # Statistics
    contingency = pd.crosstab(cluster_labels, y_map)
    chi2_stat, p_val, dof, _ = chi2_contingency(contingency)
    
    # ── Layout Setup ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12), dpi=IMAGE_DPI)
    
    gs = gridspec.GridSpec(
        3, 2, 
        figure=fig, 
        height_ratios=[6, 0.8, 2.5], 
        width_ratios=[pca_width_ratio, 1.0], 
        wspace=0.15, 
        hspace=0.3
    )
    
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[1, 0]) 
    ax_chi = fig.add_subplot(gs[1, 1]) 
    ax_tab = fig.add_subplot(gs[2, :]) 
    
    ax_leg.axis('off')
    ax_chi.axis('off')
    ax_tab.axis('off')

    # ────────────────────────── PANEL A: PCA PLOT ──────────────────────────
    for lbl in ["Alpha", "Beta"]:
        m = df["Label"] == lbl
        ax_pca.scatter(df.loc[m, "PC1"], df.loc[m, "PC2"], color=C_CLASS[lbl], alpha=0.10, s=12, rasterized=True, zorder=1)

    for lbl in ["Alpha", "Beta"]:
        m = df["Label"] == lbl
        x, y = df.loc[m, "PC1"].values, df.loc[m, "PC2"].values
        _confidence_ellipse(x, y, ax_pca, n_std=2.0, edgecolor=C_CLASS[lbl], facecolor='none', linewidth=3.0, linestyle="--", zorder=4)
        ax_pca.scatter(x.mean(), y.mean(), color=C_CLASS[lbl], marker="^", s=400, edgecolors="black", linewidths=1.5, zorder=6)

    for cid in range(3):
        m = df["Cluster"] == cid
        x, y = df.loc[m, "PC1"].values, df.loc[m, "PC2"].values
        _confidence_ellipse(x, y, ax_pca, n_std=2.0, facecolor=C_CLUST[cid], edgecolor=C_CLUST[cid], alpha=0.15, linewidth=2.0, linestyle="-", zorder=3)
        ax_pca.scatter(x.mean(), y.mean(), color=C_CLUST[cid], marker="o", s=250, edgecolors="black", linewidths=1.5, zorder=5)

    ev = pca.explained_variance_ratio_
    ax_pca.axhline(0, color="grey", lw=0.5, alpha=0.5, zorder=0)
    ax_pca.axvline(0, color="grey", lw=0.5, alpha=0.5, zorder=0)
    ax_pca.set_xlabel(f"PC1 ({ev[0]:.1%} variance)", fontsize=label_fontsize)
    ax_pca.set_ylabel(f"PC2 ({ev[1]:.1%} variance)", fontsize=label_fontsize)
    ax_pca.set_title("A.    Unified PCA Space (Centroids & 2σ Variance)", fontsize=title_fontsize, fontweight="bold", pad=15)
    ax_pca.tick_params(labelsize=tick_fontsize)

    # ────────────────────────── PANEL B: BAR PLOT ──────────────────────────
    props  = pd.crosstab(df["Cluster"], df["Label"], normalize="index") * 100
    counts = pd.crosstab(df["Cluster"], df["Label"])
    
    x_pos = np.arange(3)
    ap = props["Alpha"].values
    bp = props["Beta"].values

    ax_bar.bar(x_pos, ap, 0.55, color=C_CLASS["Alpha"], edgecolor="black", linewidth=0.8)
    ax_bar.bar(x_pos, bp, 0.55, color=C_CLASS["Beta"], edgecolor="black", linewidth=0.8, bottom=ap)

    for i, (a, b) in enumerate(zip(ap, bp)):
        if a > 5: ax_bar.text(x_pos[i], a / 2, f"{a:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=annot_fontsize)
        if b > 5: ax_bar.text(x_pos[i], a + b / 2, f"{b:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=annot_fontsize)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"Cluster {i}\n(n={counts.loc[i].sum()})" for i in range(3)], fontsize=label_fontsize)
    ax_bar.set_title("B.    GMM Cluster Enrichment", fontsize=title_fontsize, fontweight="bold", pad=15)
    ax_bar.set_ylabel("Cell Proportion (%)", fontsize=label_fontsize)
    ax_bar.set_ylim(0, 105)
    ax_bar.tick_params(axis="y", labelsize=tick_fontsize)

    for ax in [ax_pca, ax_bar]:
        for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]: ax.spines[spine].set_linewidth(1.5)

    # ────────────────────────── LEGEND & CHI-SQUARE ──────────────────────────
    legend_handles = []
    for lbl in ["Alpha", "Beta"]:
        legend_handles.append(Line2D([0], [0], color=C_CLASS[lbl], marker='^', linestyle='--', markersize=12, linewidth=2.5, markeredgecolor='k', label=f"True {lbl}"))
    for i in range(3):
        legend_handles.append(Line2D([0], [0], color=C_CLUST[i], marker='o', linestyle='-', markersize=10, linewidth=2.5, markeredgecolor='k', alpha=0.8, label=f"Cluster {i}"))

    ax_leg.legend(handles=legend_handles, loc='center', ncol=3, frameon=False, fontsize=legend_fontsize, handlelength=3.0)

    # Clean Chi-Square text (No bbox)
    p_str = r"$p$ < 0.001" if p_val < 0.001 else rf"$p$ = {p_val:.3e}"
    chi_text = rf"$\mathbf{{\chi^2\ Independence\ Test}}$" + "\n" + rf"Statistic: {chi2_stat:.1f}  |  {p_str}"
    ax_chi.text(0.5, 0.5, chi_text, transform=ax_chi.transAxes, ha="center", va="center", fontsize=annot_fontsize)

    # ────────────────────────── APA THREE-LINE TABLE ───────────────────────
    table_data = []
    
    for lbl in ["Alpha", "Beta"]:
        m = df["Label"] == lbl
        x, y = df.loc[m, "PC1"].values, df.loc[m, "PC2"].values
        table_data.append([f"True {lbl} Class", f"{m.sum()}", f"({x.mean():.2f}, {y.mean():.2f})", f"{np.var(x, ddof=1):.2f}", f"{np.var(y, ddof=1):.2f}"])
        
    for cid in range(3):
        m = df["Cluster"] == cid
        x, y = df.loc[m, "PC1"].values, df.loc[m, "PC2"].values
        table_data.append([f"Cluster {cid}", f"{m.sum()}", f"({x.mean():.2f}, {y.mean():.2f})", f"{np.var(x, ddof=1):.2f}", f"{np.var(y, ddof=1):.2f}"])

    col_headers = ["Population / Group", "Count (N)", "Centroid (PC1, PC2)", "PC1 Variance", "PC2 Variance"]

    table = ax_tab.table(cellText=table_data, colLabels=col_headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)
    table.scale(1, 2.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_text_props(fontfamily='serif')
        if row == 0:
            cell.set_text_props(weight='bold', fontfamily='serif')
            cell.visible_edges = 'BT'
            cell.set_linewidth(2.0)
        elif row == len(table_data):
            cell.visible_edges = 'B'
            cell.set_linewidth(2.0)
        else:
            cell.visible_edges = ''

    # ── Save / show ────────────────────────────────────────────────────────
    plt.subplots_adjust(bottom=0.05)
    
    if save and output_dir:
        for ext in ("png", "pdf"):
            p = output_dir / f"GMM_Unified_Summary_Final.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=IMAGE_DPI)
        print(f"✅ Final Master Figure saved → {output_dir}")

    if show:
        plt.show()
    plt.close(fig)


def make_figure2(
    features_df: pd.DataFrame, cluster_labels: np.ndarray, pca, output_dir=None,
    top_n: int = 4, title_fontsize: int = 15, label_fontsize: int = 13,
    tick_fontsize: int = 11, show: bool = True, save: bool = True,
):
    print("\n" + "=" * 50)
    print("Generating Figure 2: Morphological Gradient KDEs")
    print("=" * 50)

    # ── 1. Apply Styles ────────────────────────────────────────────────────
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set_context("paper")
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': label_fontsize,
    })
    
    # Internal aliases to ensure we don't rely on global scope being identical
    C_CLASS = {"Alpha": CLASS_COLORS["Alpha"], "Beta": CLASS_COLORS["Beta"]}
    C_CLUST = C_CLUST_GLOBAL
    L_CLUST = L_CLUST_GLOBAL
    
    # ── Identify top PC1 drivers ───────────────────────────────────────────
    X_feat = features_df.drop(columns=["label", "Cluster", "PC1_Score"], errors="ignore")
    loadings = pd.Series(pca.components_[0], index=X_feat.columns)
    drivers = loadings.abs().sort_values(ascending=False).head(top_n).index.tolist()

    # ── Layout ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, top_n, figsize=(4.5 * top_n, 9.8), dpi=IMAGE_DPI, gridspec_kw=dict(hspace=0.75, wspace=0.28))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.22)

    # Row A: Cell-type KDEs
    for col, feat in enumerate(drivers):
        ax = axes[0, col]
        for lbl_val, lbl_name in [(0, "Alpha"), (1, "Beta")]:
            data = features_df.loc[features_df["label"] == lbl_val, feat].values
            sns.kdeplot(data, ax=ax, fill=True, color=C_CLASS[lbl_name], alpha=0.5, linewidth=1.8)
        ax.set_title(feat.replace("_", " "), fontweight="bold")

    # Row B: Cluster KDEs
    for col, feat in enumerate(drivers):
        ax = axes[1, col]
        for cid in range(3):
            data = features_df.loc[cluster_labels == cid, feat].values
            sns.kdeplot(data, ax=ax, fill=True, color=C_CLUST[cid], alpha=0.48, linewidth=1.8)
        ax.set_xlabel("Normalised value")

    # ── Spread Legend ──────────────────────────────────────────────────────
    legend_handles = []
    legend_handles.append(Line2D([0], [0], color=C_CLASS["Alpha"], lw=4, alpha=0.6, label="Alpha"))
    legend_handles.append(Line2D([0], [0], color=C_CLASS["Beta"], lw=4, alpha=0.6, label="Beta"))
    legend_handles.append(Line2D([0], [0], color='none', label="   |   "))
    
    for i in range(3):
        # Index error fix: using L_CLUST list (length 3) with range(3)
        legend_handles.append(Line2D([0], [0], color=C_CLUST[i], lw=4, alpha=0.6, label=f"Cluster {i} ({L_CLUST[i]})"))

    fig.legend(handles=legend_handles, loc='lower center', ncol=6, frameon=False, 
               fontsize=label_fontsize, handlelength=2.5, bbox_to_anchor=(0.5, 0.01))

    if save and output_dir: fig.savefig(output_dir / "GMM_Fig2_Gradient.png", bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

