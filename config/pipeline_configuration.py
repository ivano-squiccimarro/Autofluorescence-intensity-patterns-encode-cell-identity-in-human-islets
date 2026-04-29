
from pathlib        import Path
from dataclasses    import dataclass, field
from typing         import Dict, List
from skopt.space    import Integer, Real, Categorical

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import (
                                RandomForestClassifier,
                                GradientBoostingClassifier,
                                ExtraTreesClassifier
                            )
from sklearn.neighbors      import KNeighborsClassifier
from lightgbm               import LGBMClassifier
from xgboost                import XGBClassifier




@dataclass(frozen=True)
class PipelineConfig:

    image_normalize:bool = False
    
    # General   
    random_seeds: List[int] = field(default_factory=lambda: range(10))
    n_splits: int = 5
    test_ratio: float = 0.2
    overlap_threshold: float = 1.1
    width:int = 150
    n_jobs: int = -1 
    variance_threshold: float = 0.0

    # Data augmentation 
    global_aug_factor: int = 200
    train_aug_factor: int = 20
    val_aug_factor: int = 5
 
    # Stress testing
    n_stress_test_levels: int = 3

    # Balancing Classes Strategies
    balance: bool = True
    balance_with_smote: bool = False
    extra_augmentation_per_class : dict = None
    
    # Train Outliers Removal
    train_outlier_removal: bool = False
    contamination: float        = 0.00 

    # Bayesian optimization
    bayesian_opt: bool = True
    n_bayesian_steps: int = 40

    # Voting & misc
    voting_method: str = 'soft'
    compression_level: int = 1
    verbosity: int = 1

    # Paths
    numpy_images_path: Path = Path('data/input/numpy files')
    metadata_path: Path = Path('data/input/db_files.csv')
    image_df_save_path: Path = Path('image_df.pkl')

    MODEL_DICT: Dict[str, object] = field(default_factory=lambda: {
        "RandomForest": RandomForestClassifier(n_jobs=-1, warm_start = False),
        "ExtraTrees": ExtraTreesClassifier(n_jobs=-1),
        "LGBM": LGBMClassifier(n_jobs=-1, verbose=-1),
        "XGBoost": XGBClassifier(n_jobs=-1, use_label_encoder=False, eval_metric='logloss', verbosity=0),
        "KNeighbors": KNeighborsClassifier(metric = 'minkowski', n_jobs=-1),
        "LogisticRegression": LogisticRegression(solver = 'saga', penalty = 'elasticnet', max_iter=5000)
    })

    model_hyperparams: Dict[str, Dict[str, object]] = field(default_factory=lambda: {
        "RandomForest": {
            'n_estimators': Integer(100, 2000),
            'max_depth': Integer(2, 100),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2']),
        },
        "ExtraTrees": {
            'n_estimators': Integer(100, 2000),
            'max_depth': Integer(2, 100),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2']),
        },
        "LGBM": {
            'n_estimators': Integer(100, 2000),
            'learning_rate': Real(0.01, 0.3, 'log-uniform'),
            'num_leaves': Integer(10, 150),
            'max_depth': Integer(5, 50),
            'reg_alpha': Real(1e-2, 10.0, 'log-uniform'), 
            'reg_lambda': Real(1e-2, 10.0, 'log-uniform'),
            'colsample_bytree': Real(0.1, 1.0, 'uniform'),
            'subsample': Real(0.1, 1.0, 'uniform'),
        },
        "XGBoost": {
            'n_estimators': Integer(100, 2000),
            'learning_rate': Real(0.01, 0.3, 'log-uniform'),
            'max_depth': Integer(5, 50),
            'subsample': Real(0.1, 1.0, 'uniform'),
            'colsample_bytree': Real(0.6, 1.0, 'uniform'),
            'alpha': Real(1e-2, 10.0, 'log-uniform'), 
            'lambda': Real(1e-2, 10.0, 'log-uniform'),
            'gamma': Real(0, 0.5, 'uniform'),
        },
        "KNeighbors": {
            'n_neighbors': Integer(1, 50),
            'weights': Categorical(['uniform', 'distance']),
            'p': Real(1, 5),
        },
        "LogisticRegression": {
            'C': Real(1e-3, 1e3, 'log-uniform'),
            'warm_start':  Categorical([True, False]),
            'l1_ratio': Real(0, 1, 'uniform'),
        },
    })

    # Meta‑models
    meta_models: Dict[str, object] = field(default_factory=lambda: {
        'Logistic': LogisticRegression(max_iter=5000),
        'KNeighbors': KNeighborsClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    })

