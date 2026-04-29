####################################################
# Dataset Augmentation Functions
####################################################

import cv2
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from math import ceil



def mirror_image(
    image: np.ndarray,
) -> np.ndarray:
    return cv2.flip(image, 1)

def rotate_image(
    image: np.ndarray,
    angle: int,
)-> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

augmentation_functions = [
    'mirror',
    'rotate'
]

def apply_random_augmentation(
    image: np.ndarray,
    seed_value: int

) -> np.ndarray:
    """
    Apply one or more random augmentations to the input image.

    Instead of selecting only one augmentation function from the global
    list 'augmentation_functions', this version randomly selects a subset
    (at least one and up to all available) and applies them sequentially.
    Each augmentation is applied at most once.

    Parameters:
        image (numpy.ndarray): The input image to augment.

    Returns:
        augmented_image (numpy.ndarray): The augmented image.
        augmentation_details (str): Description of the augmentations applied.
    """
    random.seed(seed_value)

    num_to_apply = random.randint(1, len(augmentation_functions))
    selected_augs = random.sample(augmentation_functions, num_to_apply)

    details = []

    for aug_type in selected_augs:
        if aug_type == 'mirror':
            image = mirror_image(image)
            details.append("mirror")
        elif aug_type == 'rotate':
            #Randomly choose an angle between 0 and 359 degrees.
            angle = random.randint(0, 359)
            image = rotate_image(image, angle)
            details.append(f"rotate_{angle}")

    augmentation_details = "_".join(details)
    return image, augmentation_details

def augment_row(
    row,
    n_augmentations: int,
    base_seed: int
  ):
    """
    Apply random augmentation n_augmentations times on a single row.
    """
    image = row['image']
    label = row['label']
    original_index = row.name
    augmented = []
    for i in range(n_augmentations):
        # Creates a unique seed based on the base_seed, row index, and augmentation iteration (i)
        row_seed = base_seed + hash(str(original_index) + str(i)) % 1000000
        aug_img, details = apply_random_augmentation(image, row_seed)
        augmented.append({
            'image': aug_img,
            'label': label,
            'augmentation': details,
            'original_index': original_index 

        })
    return augmented


def create_random_augmented_dataset(
    train_df: pd.DataFrame,
    n_augmentations: int = 1,
    balance: bool = True,
    extra_augment_pct: dict = {0: 0.3},
    random_state: int | None = None,
    _original_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Augment dataset with options to balance classes and add extra augmentation.
    This version ensures augmentations are always created from the original source images
    and correctly implements the sequential goal: Baseline N-Augmentation THEN Balancing.
    """
    # seed
    base_seed = random_state if random_state is not None else 42

    # --- Preserve the original, clean dataset ---
    if _original_df is None:
        # Save a clean copy of the original data, ensuring the index is unique
        _original_df = train_df.copy().reset_index(drop=True).set_index(pd.Index(range(len(train_df))))

    # simple augmentation (no balancing) - used for internal recursive calls
    if not balance:
        if n_augmentations < 0:
            raise ValueError("n_augmentations must be non-negative.")
        if n_augmentations == 0:
            return pd.DataFrame([], columns=['image','label','augmentation'])
        
        # NOTE: train_df here is already a subset from the original in the recursive call
        train_df_with_index = train_df.copy() # Use existing index for augment_row
        
        rows = Parallel(n_jobs=-1)(
            delayed(augment_row)(row, n_augmentations, base_seed)
            for _, row in train_df_with_index.iterrows()
        )
        flat = [r for sub in rows for r in sub]

        if not flat:
            return pd.DataFrame([], columns=['image','label','augmentation'])
        
        # Create DataFrame from flat list
        aug_df = pd.DataFrame(flat)
        # Assuming augment_row uses the current DataFrame index as 'original_index'
        aug_df = aug_df.set_index('original_index')
        return aug_df

    # balancing path (where balance=True)
    counts = _original_df['label'].value_counts()
    parts = [] # We will build the new balanced dataset here

    # 1. First, calculate the total size required for the majority class after n_augmentations
    majority_label = counts.index[0] # Assumes counts is sorted by default (largest first)
    majority_count = counts.iloc[0]
    
    # The new size of the majority class after applying N augmentations is the target balance point
    # majority_count originals + (majority_count * n_augmentations) augs
    target_max_size = majority_count * (n_augmentations + 1)
    
    # 2. Iterate through all classes to determine their augmentation needs
    for lbl, cnt in counts.items():
        subset = _original_df[_original_df['label'] == lbl]
        
        # Calculate the size this class will reach by just applying the baseline N augmentations
        size_after_n_aug = cnt * (n_augmentations + 1)

        # Determine the final total count goal for this class
        if lbl == majority_label:
            # Majority class just gets the baseline N augmentations
            final_target_total_count = target_max_size
        else:
            # Minority class target is the MAX of (its n-aug size) and (the majority's n-aug size)
            final_target_total_count = max(size_after_n_aug, target_max_size)

        # The number of *new augmentations* we need to create
        new_augmentations_needed = final_target_total_count - cnt
        
        if new_augmentations_needed <= 0:
            # If the class size after N-aug already meets or exceeds the max target (unlikely), skip
            parts.append(subset) # Add original data only
            continue

        # Calculate the number of augmentations per image
        per_img = ceil(new_augmentations_needed / cnt)
        
        # Recursive call to create the augmentations (balance=False, draws from original subset)
        aug_data = create_random_augmented_dataset(
            subset, n_augmentations=per_img, balance=False,
            random_state=random_state, _original_df=_original_df
        )
        
        # Downsample if we overshot the target
        if len(aug_data) > new_augmentations_needed:
            aug_data = aug_data.sample(n=new_augmentations_needed, random_state=random_state)
            
        # Add the original images and the new augmentations to the parts list
        parts.append(subset.reset_index(drop=True))
        parts.append(aug_data.reset_index(drop=True))

    balanced_df = pd.concat(parts, ignore_index=True)
    
    # Set the 'augmentation' column correctly for original images
    balanced_df['augmentation'] = balanced_df['augmentation'].apply(
        lambda x: x if isinstance(x, str) else 'Original' 
    )

    # --- Extra augmentation block ---
    if extra_augment_pct:
        extras = []
        for lbl, pct in extra_augment_pct.items():
            if pct <= 0 or lbl not in _original_df['label'].unique():
                continue
            
            # Augmentation must come from the _original_df
            subset = _original_df[_original_df['label'] == lbl]
            base = len(subset)
            extra = int(base * pct)

            if extra <= 0:
                continue

            per_img = ceil(extra / base)
            aug_extra = create_random_augmented_dataset(
                subset, n_augmentations=per_img, balance=False,
                random_state=random_state, _original_df=_original_df
            )
            
            if len(aug_extra) > extra:
                aug_extra = aug_extra.sample(n=extra, random_state=random_state).reset_index(drop=True)
            
            # Mark these specifically as extra augmentations
            aug_extra['augmentation'] = aug_extra['augmentation'].apply(lambda x: f"Extra_{x}" if isinstance(x, str) else 'ExtraAugmented')
            extras.append(aug_extra)
        
        final = pd.concat([balanced_df] + extras, ignore_index=True)
    else:
        final = balanced_df

    return final

