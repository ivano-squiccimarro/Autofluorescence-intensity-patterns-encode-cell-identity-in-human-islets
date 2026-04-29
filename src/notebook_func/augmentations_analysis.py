
import numpy    as np
import pandas   as pd
import seaborn  as sns

from pathlib import Path

import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.ticker    as mtick
from matplotlib.ticker      import MultipleLocator

from tqdm.auto  import tqdm


from sklearn.metrics.pairwise import cosine_similarity

# Local Imports
from src.feature_extraction_functions import extract_feats
from src.dataset_augmentation_functions import rotate_image
from config.presentation_config import set_nature_style, TECH_TO_PUB, IMAGE_DPI

def run_augmentation_regime_analysis(
    image_df: pd.DataFrame,
    output_dir: Path,
    n_samples: int = 100,
    angles: range = range(0, 181, 1),
    tech_names: list = None,
    save: bool = True,
    show: bool = False
):
    """
    Analyzes how image rotation affects feature stability and generates 
    Nature-style plots for the augmentation justification.
    """
    set_nature_style()
    SKYBLUE_COLOR = '#87CEEB'
    EPSILON = 1e-8

    # 1. Sampling 
    if len(image_df) < n_samples:
        n_samples = len(image_df)
    sample_df = image_df.sample(n_samples, random_state=42)

    # 2. Generating Image Configurations [cite: 563]
    data_for_extraction = []
    for image_id, row in tqdm(sample_df.iterrows(), total=n_samples, desc="Rotating Images"):
        for angle in angles:
            img = rotate_image(row['image'], angle=angle) if angle != 0 else row['image']
            data_for_extraction.append({
                'image_id': image_id, 'angle': angle, 'image': img, 'label': row['label']
            })

    extraction_input_df = pd.DataFrame(data_for_extraction).set_index(['image_id', 'angle'])

    # 3. Feature Extraction [cite: 564]
    all_features_df = extract_feats(extraction_input_df[['image', 'label']], n_jobs=-1, verbosity=0)
    
    # Apply Technical Naming
    if tech_names and len(all_features_df.columns) == len(tech_names) + 1:
        all_features_df.columns = tech_names + ['label']

    # 4. Process Deviations 
    feature_cols = all_features_df.drop('label', axis=1).columns.tolist()
    features_orig = all_features_df.xs(0, level='angle')[feature_cols]
    
    deviations, rel_diff_vectors = [], []

    for img_id in tqdm(sample_df.index, desc="Calculating Metrics"):
        orig_vec = features_orig.loc[img_id].to_numpy().reshape(1, -1)
        rotated_df = all_features_df.loc[img_id][feature_cols]
        
        for angle, row in rotated_df.iterrows():
            rot_vec = row.to_numpy().reshape(1, -1)
            sim = cosine_similarity(orig_vec, rot_vec)[0, 0]
            deviations.append({'image_id': img_id, 'angle': angle, 'cosine_dist': 1.0 - sim})
            
            if angle != 0:
                rel_diff_vectors.append(np.abs(orig_vec.flatten() - rot_vec.flatten()) / (np.abs(orig_vec.flatten()) + EPSILON))

    # 5. Visualization [cite: 568, 569]
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.4)
    
    # Panel A: Cosine Distance
    ax1 = fig.add_subplot(gs[0])
    dev_df = pd.DataFrame(deviations)
    sns.lineplot(data=dev_df, x='angle', y='cosine_dist', ax=ax1, errorbar='se', color=SKYBLUE_COLOR, linewidth=2.5)
    ax1.set_title("A. Feature Vector Similarity vs. Rotation Angle", fontweight='bold', loc='left', pad=20)
    ax1.set_ylabel("Cosine Distance\n(1 - Similarity)")
    ax1.xaxis.set_major_locator(MultipleLocator(30))

    # Panel B: Top Affected Features [cite: 567, 570]
    ax2 = fig.add_subplot(gs[1])
    rel_df = pd.DataFrame(np.vstack(rel_diff_vectors), columns=feature_cols)
    long_df = rel_df.melt(var_name='Technical Name', value_name='Error')
    long_df['Publication Name'] = long_df['Technical Name'].map(TECH_TO_PUB)
    
    top_feats = long_df.groupby('Publication Name')['Error'].mean().sort_values(ascending=False).head(10).index
    sns.barplot(data=long_df[long_df['Publication Name'].isin(top_feats)], x='Error', y='Publication Name', 
                color=SKYBLUE_COLOR, order=top_feats, errorbar='se', ax=ax2)
    ax2.set_title("B. Top 10 Features Most Affected by Rotation", fontweight='bold', loc='left', pad=20)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "Rotation_Stability_Analysis.png", dpi=IMAGE_DPI, bbox_inches='tight')
    
    if show: plt.show()
    plt.close()