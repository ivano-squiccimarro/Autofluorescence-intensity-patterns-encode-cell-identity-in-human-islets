import pandas               as pd
import matplotlib.pyplot    as plt
from pathlib                import Path

from config.presentation_config import IMAGE_DPI

def create_journal_hyperparameter_table(
        first_row_text_size=16,
        text_size=14,
        first_columns_text_size=14,
        figshape=(12, 14),
        save_=True,
        show_=False,
        outputdir=None
    ):
    # 1. Data Definition
    raw_data = [
        ("Random Forest", "n_estimators", "100 – 2000", "Integer"),
        ("Random Forest", "max_depth", "2 – 100", "Integer"),
        ("Random Forest", "min_samples_split", "2 – 20", "Integer"),
        ("Random Forest", "min_samples_leaf", "1 – 10", "Integer"),
        ("Random Forest", "max_features", "{sqrt, log2}", "Categorical"),
        ("Extra Trees", "n_estimators", "100 – 2000", "Integer"),
        ("Extra Trees", "max_depth", "2 – 100", "Integer"),
        ("Extra Trees", "min_samples_split", "2 – 20", "Integer"),
        ("Extra Trees", "min_samples_leaf", "1 – 10", "Integer"),
        ("Extra Trees", "max_features", "{sqrt, log2}", "Categorical"),
        ("LightGBM", "n_estimators", "100 – 2000", "Integer"),
        ("LightGBM", "learning_rate", "0.01 – 0.3", "Log-Uniform"),
        ("LightGBM", "num_leaves", "10 – 150", "Integer"),
        ("LightGBM", "max_depth", "5 – 50", "Integer"),
        ("LightGBM", "reg_alpha", "0.01 – 10.0", "Log-Uniform"),
        ("LightGBM", "reg_lambda", "0.01 – 10.0", "Log-Uniform"),
        ("LightGBM", "colsample_bytree", "0.1 – 1.0", "Uniform"),
        ("LightGBM", "subsample", "0.1 – 1.0", "Uniform"),
        ("XGBoost", "n_estimators", "100 – 2000", "Integer"),
        ("XGBoost", "learning_rate", "0.01 – 0.3", "Log-Uniform"),
        ("XGBoost", "max_depth", "5 – 50", "Integer"),
        ("XGBoost", "subsample", "0.1 – 1.0", "Uniform"),
        ("XGBoost", "colsample_bytree", "0.6 – 1.0", "Uniform"),
        ("XGBoost", "alpha", "0.01 – 10.0", "Log-Uniform"),
        ("XGBoost", "lambda", "0.01 – 10.0", "Log-Uniform"),
        ("XGBoost", "gamma", "0 – 0.5", "Uniform"),
        ("K-Neighbors", "n_neighbors", "1 – 50", "Integer"),
        ("K-Neighbors", "weights", "{uniform, distance}", "Categorical"),
        ("K-Neighbors", "p", "1 – 5", "Uniform"),
        ("Logistic Reg.", "C", "0.001 – 1000", "Log-Uniform"),
        ("Logistic Reg.", "l1_ratio", "0 – 1", "Uniform"),
        ("Logistic Reg.", "warm_start", "{True, False}", "Categorical"),
    ]

    df = pd.DataFrame(raw_data, columns=["Algorithm", "Hyperparameter", "Range / Set", "Distribution"])

    fig, ax = plt.subplots(figsize=figshape)
    ax.axis('off')

    # 2. Draw Table (bbox adjusted for top/bottom line space)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='left', 
        loc='center',
        bbox=[0, 0, 1, 0.95]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(text_size)

    # 3. Journal Styling Logic
    cells = table.get_celld()
    current_algo = None
    group_boundaries = []

    for row_idx in range(len(df) + 1):
        for col_idx in range(len(df.columns)):
            cell = cells[(row_idx, col_idx)]
            cell.set_edgecolor('white')  # Hide all internal borders
            cell.set_linewidth(0)
            
            # --- Header Styling ---
            if row_idx == 0:
                cell.set_text_props(weight='bold', size=first_row_text_size)
                cell.set_facecolor('none')
            
            # --- Data Cell Alignment & Algorithm Merging ---
            else:
                cell.set_facecolor('none')
                if col_idx == 0:
                    algo_name = df.iloc[row_idx-1, 0]
                    # Only show algorithm name once per group
                    if algo_name != current_algo:
                        cell.set_text_props(weight='bold', size=first_columns_text_size)
                        current_algo = algo_name
                        if row_idx > 1:
                            group_boundaries.append(row_idx)
                    else:
                        cell.get_text().set_text("")

    # 4. Manual Rule Placement (The Journal Look)
    # 0.95 is the top of the table bbox
    row_height = 0.95 / (len(df) + 1)
    
    # Top and Header Rule
    ax.plot([0, 1], [0.95, 0.95], color='black', lw=2.5, transform=ax.transAxes) # Top line
    ax.plot([0, 1], [0.95 - row_height, 0.95 - row_height], color='black', lw=1.5, transform=ax.transAxes) # Under header
    
    # Bottom Rule
    ax.plot([0, 1], [0, 0], color='black', lw=2.5, transform=ax.transAxes) 
    
    # Thin separators for Algorithm groups
    for boundary in group_boundaries:
        y_pos = 0.95 - (boundary * row_height)
        ax.plot([0, 1], [y_pos, y_pos], color='#dddddd', lw=0.8, transform=ax.transAxes)

    if save_ and outputdir:
        outputdir = Path(outputdir)
        outputdir.mkdir(parents=True, exist_ok=True)
        savepath = outputdir / Path("Hyperparameter_Search_Space_Journal.png")
        plt.savefig(savepath, dpi=IMAGE_DPI, bbox_inches='tight')
    if show_:
        plt.show()
    plt.close()

