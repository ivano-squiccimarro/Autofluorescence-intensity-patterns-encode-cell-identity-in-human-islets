# Autofluorescence intensity patterns encode α/β cell identity in human islets

## Introduction
Understanding the behavior of Alpha and Beta cells within intact human islets is essential for elucidating mechanisms of metabolic control in Diabetes. Current cell-type identification strategies rely on destructive labeling or on advanced imaging modalities such as Fluorescence Lifetime Imaging Microscopy (FLIM), which provide rich metabolic information but require specialized instrumentation and acquisition protocols.

This project demonstrates that structured intracellular intensity patterns derived from endogenous autofluorescence are sufficient to discriminate Alpha and Beta cells in living human islets. Using rotation-invariant Local Ternary Pattern (LTP) descriptors combined with morphological features, we achieve highly accurate classification (**AUC = 0.92**), surpassing previously reported benchmarks.

Interpretability analyses demonstrate that discrimination is driven predominantly by fine-scale intracellular intensity organization rather than global morphology. In the spectral window employed, cytoplasmic autofluorescence is prominently shaped by lipofuscin-rich granules. Consistent with higher lipofuscin accumulation in Beta cells, the dominant features identified here likely reflect differences in granule abundance and spatial organization between endocrine cell types. This provides a biologically grounded and fully non-destructive framework for the identification of pancreatic islet cell types.

## 🧬 Project Highlights
* **Non-Destructive Discrimination (AUC = 0.92):** Surpasses reported benchmarks by identifying endogenous intensity patterns that encode sufficient structural information for reliable $\alpha$/$\beta$ discrimination, operating entirely independently of external patient-linked metadata. 
  
* **Robustness to common Geometric Variations:** The classification pipeline is specifically engineered to remain robust against rotation and mirroring, the most frequent geometric transformations encountered during microscopy acquisition. Stability analysis confirms that the use of rotation-invariant descriptors ensures reliable performance regardless of the cell's orientation or lateral flip.

* **Intracellular "Granule" Logic:** Interpretability tools (SHAP/PDP) prove that classification is driven by fine-scale intensity organization—likely lipofuscin-rich granules—rather than simple cell shape.

* **Real-Time Engine:** A refactored, parallelized prediction pipeline utilizing `numba` and in-place memory views, reducing serial latency from ~18 ms to **~4.8 ms** (~210 FPS) with zero numerical degradation.

---

## 📂 Repository Structure
The codebase is strictly separated into frontend execution (Jupyter Notebooks) and backend logic (`src/`), ensuring clean, readable, and reproducible experiments.

```text
├── config
│   ├── __init__.py
│   ├── pipeline_configuration.py
│   ├── presentation_config.py
│   └── speed_benchmark_config.py
├── data
│   └── input
│       ├── db_files.csv
│       └── Fabio_rowstodrop.csv
├── notebook
│   └── Analysis_of Results_and_Graph_and_Tables_Generator.ipynb
├── src
│   ├── notebook_func
│   │   ├── __init__.py
│   │   ├── augmentations_analysis.py
│   │   ├── cross_dataset_analysis.py
│   │   ├── eda_visualization_functions.py
│   │   ├── hyperparameter_models_table.py
│   │   ├── model_interpretability_funcs.py
│   │   ├── model_performance_evaluation.py
│   │   ├── realtime_performance.py
│   │   └── results_retrieval.py
│   ├── __init__.py
│   ├── bayesian_hyperparameter_search_functions.py
│   ├── dataset_augmentation_functions.py
│   ├── feature_extraction_functions.py
│   ├── final_training_and_testing_evaluation_functions.py
│   ├── from_images_collection_to_dataframe.py
│   ├── Original_RealTimeFrameClassification.py
│   ├── per_seed_directory_creation_and_inner_cross_validaiton_setup.py
│   ├── pipeline_seed.py
│   ├── stacking_ensemble_functions.py
│   └── voting_ensemble_functions.py
├── .gitignore
├── main.py
├── requirements.txt
└── smart_tree_creator.py
```

## 🚀 Installation & Setup
**1. Clone the repository:**
```text
git clone [https://github.com/ivano-squiccimarro/Autofluorescence-intensity-patterns-encode-cell-identity-in-human-islets.git](https://github.com/ivano-squiccimarro/Autofluorescence-intensity-patterns-encode-cell-identity-in-human-islets.git)
cd Autofluorescence-intensity-patterns-encode-cell-identity-in-human-islets
```
**2. Create a virtual environment:**
```text
python -m venv venv
.\venv\Scripts\Activate.ps1
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
**3. Install dependencies:**
```text
pip install -r requirements.txt
```


## 🚀Reproducing Experimental Results
**1. Main Experiment**
To replicate the main experiment results, execute the pipeline as configured:
```text
python main.py
```

**2. S2.5: Importance of Augmentation Regime**
To replicate the control experiments proving the necessity of rotation-based augmentation:
  1. Open config/pipeline_configurations.py and set:
```text
  global_aug_factor: int = 0
  train_aug_factor: int = 0
  val_aug_factor: int = 0
 ```
  2. Open main.py and change the output destination:
 ```text
 base_results = Path("Dataset++_No_Aug")
 ```

**3. S2.6: Inclusion of Extended Alpha Cell Data**
To produce the data configurations for cross-dataset generalizability tests, update config/pipeline_configurations.py and main.py according

  1. For No Extra Alpha Cells in dataset and Balancing with SMOTE ( named in the article 'Dataset_SMOTE') go in pipeline_configurations.py and set
```text
      balance: bool = True
      balance_with_smote: bool = True
```
Then go to main.py and set name of the folder results as:
```text
base_results = Path("Dataset_SMOTE")
```
  2. For No Extra Alpha Cells in dataset and No Balancing ( named in the article 'Dataset') go in pipeline_configurations.py and set
```text
    balance: bool = False
    balance_with_smote: bool = False
```
Then go to main.py and set name of the folder results as:
 ```text
 base_results = Path("Dataset")
 ```
Here the logic table of the Experiments:

| Configuration | balance | balance_with_smote | Output in main.py       |
| ------------- | ------- | ------------------ | ----------------------- |
| Dataset++     | True    | False              | Path(""Dataset++"")     |
| Dataset       | False   | False              | Path(""Dataset"")       |
| Dataset_SMOTE | True    | True               | Path(""Dataset_SMOTE"") |

## 📊 Running the Analysis 
The analysis is driven by the Jupyter Notebook in /notebooks. Simply launch Jupyter and run the cells sequentially. All heavy statistical processing, caching of SHAP values, and Nature-style rendering are handled automatically by the modular backend in /src/notebook_func/.

