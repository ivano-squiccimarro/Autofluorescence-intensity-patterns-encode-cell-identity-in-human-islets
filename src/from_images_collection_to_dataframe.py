
############################################################
# Utility Functions to select and process input numpy images 
# in a confortable format and create a dataframe 
############################################################


import os
import cv2
import joblib
import logging
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from pathlib import Path
from typing import List,Tuple




def crop_cells(
    image: np.ndarray,
    contours: List[np.ndarray]
) -> List[np.ndarray]:

    """Crop images of cells based on contours."""
    cropped_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Avoid extremely small contours (noise)
        if w > 5 and h > 5:
            cropped_image = image[y:y+h, x:x+w]
            cropped_images.append(cropped_image)
    return cropped_images

@njit
def compute_overlap_area(
    image1: np.ndarray,
    image2: np.ndarray
  ) -> int:
    """
    Compute the elementwise overlap area where both image1 and image2 have positive values.
    """
    overlap = (image1 > 0) & (image2 > 0)
    return overlap.sum()

def detect_overlaps(
    metadata: pd.DataFrame,
    image_dir: str,
    threshold: float = 0.1,
    verbosity: int = 1
) -> set:
    """
    Detect overlapping images within each group (grouped by 'date', 'islet', 'glucose')
    and return a set of filenames to drop.
    """
    if verbosity > 0:
        logging.info("Detecting overlapping images...")

    # Group by the specified keys
    grouped_metadata = metadata.groupby(['date', 'islet', 'glucose'])
    overlapping_files = set()

    for group_keys, group in tqdm(grouped_metadata, desc="Checking Overlaps"):
        donor, islet, glucose = group_keys
        group_id = f"Donor: {donor}, Islet: {islet}, Glucose: {glucose}"
        if verbosity > 0:
            logging.info(f"Processing group: {group_id}")

        # Pre-load images and compute their positive pixel counts once
        all_cells = []
        for _, row in group.iterrows():
            file_name = row['name of file']
            if not file_name.endswith('.npy'):
                file_name += '.npy'
            image_path = os.path.join(image_dir, file_name)
            image = np.load(image_path)
            pos_area = np.count_nonzero(image > 0)
            all_cells.append((file_name, image, pos_area))

        # Loop over unique pairs of images in the group using itertools.combinations
        for (file1, image1, area1), (file2, image2, area2) in itertools.combinations(all_cells, 2):
            overlap_area = compute_overlap_area(image1, image2)
            # Check if the overlap area exceeds the threshold based on either image's area
            if (area1 > 0 and overlap_area >= threshold * area1) or (area2 > 0 and overlap_area >= threshold * area2):
                if verbosity > 0:
                    logging.info(f"Overlap detected between {file1} and {file2} (Overlap Area: {overlap_area})")
                overlapping_files.update([file1, file2])

    if verbosity > 0:
        logging.info(f"Total overlapping files detected: {len(overlapping_files)}")
    return overlapping_files

def drop_rows(
    filtered_metadata: pd.DataFrame,
    path_todrop: str,
    retention: str = 'balanced'
) -> pd.DataFrame:
    """
    Drop rows from filtered_metadata based on the entries in the path_todrop file.

    Modes:
    - 'Total': Do not drop any rows.
    - 'Fabio': Drop all rows found in the CSV regardless of cell type.
    - 'balanced': Drop rows specified in the CSV except that rows with cell type "alpha" are preserved.

    Parameters:
    - filtered_metadata (pd.DataFrame): Metadata DataFrame to be filtered.
    - path_todrop (str): Path to the CSV file containing rows to drop.
    - retention (str): Retention mode of images ('Total', 'Fabio', or 'balanced').

    Returns:
    - filtered_metadata (pd.DataFrame): Updated metadata with the specified rows dropped.
    """
    KNOWN_BAD_FILES = []
    
    # Validate mod value
    retention = retention.lower()
    logging.info(retention)
    if retention not in ['total', 'fabio', 'balanced']:
        raise ValueError("retention must be 'Total', 'Fabio', or 'balanced'.")

    # If mod is 'total', do not drop any rows.
    if retention == 'total':
        file_ids = filtered_metadata['R64 file'].apply(lambda x: x.split('\\')[-1])
        filtered_metadata = filtered_metadata[~file_ids.isin(KNOWN_BAD_FILES)]
        logging.info(f"retention set to Total: All images are maintend except for {KNOWN_BAD_FILES}.")
        return filtered_metadata
   
    # Load rows to drop
    df_todrop = pd.read_csv(path_todrop)

    # Ensure 'date' is a string in both DataFrames
    df_todrop['date'] = df_todrop['date'].astype(str)
    filtered_metadata['date'] = filtered_metadata['date'].astype(str)

    # Get unique combinations of date and islet present in both DataFrames
    common_dates_islets = set(zip(df_todrop['date'], df_todrop['islet'])) & set(zip(filtered_metadata['date'], filtered_metadata['islet']))

    # Initialize lists to track dropped indices and unmatched rows
    dropped_indices = []
    unmatched_rows = []

    # Iterate through each common date and islet pair
    for date, islet in common_dates_islets:
        # Subset for the specific date and islet
        df_todrop_subset = df_todrop[(df_todrop['date'] == date) & (df_todrop['islet'] == islet)]
        filtered_metadata_subset = filtered_metadata[(filtered_metadata['date'] == date) & (filtered_metadata['islet'] == islet)]

        transformed_files = filtered_metadata_subset['R64 file'].apply(lambda x: x.split('\\')[-1].split('.')[0]).tolist()

        # Determine the maximum numbered entries for 'alpha2_' and 'beta2_'
        alpha2_entries = (
            np.max([int(file.split('_')[-1]) for file in transformed_files if "alpha2_" in file])
            if any("alpha2_" in file for file in transformed_files) else 0
        )
        beta2_entries = (
            np.max([int(file.split('_')[-1]) for file in transformed_files if "beta2_" in file])
            if any("beta2_" in file for file in transformed_files) else 0
        )
        # Iterate through each row in df_todrop_subset
        for i in range(df_todrop_subset.shape[0]):
            cell_type = df_todrop_subset['cell_type'].iloc[i]

            # For 'balanced' mode, skip dropping rows with cell_type "alpha"
            if (cell_type == 'alpha') and (retention == 'balanced'):
                continue

            glucose = df_todrop_subset['glucose'].iloc[i].split('m')[0]
            cell_number = df_todrop_subset['cell_number'].iloc[i] + 1

            # Adjust cell_number for glucose of 16 based on alpha2_ or beta2_ entries
            if glucose == '16':
                if cell_type == 'alpha':
                    cell_number = str(cell_number - alpha2_entries)
                else:
                    cell_number = str(cell_number - beta2_entries)
            else:
                cell_number = str(cell_number)

            # Construct the reference based on the modified cell number
            reference = f'{cell_type}{glucose}_{cell_number}'

            # Find rows in the metadata subset that match the reference
            matching_rows = filtered_metadata_subset[
                filtered_metadata_subset['R64 file'].apply(
                    lambda x: x.split('\\')[-1].split('.')[0]
                ) == reference
            ]

            # Record indices of matching rows for dropping or log unmatched rows
            if not matching_rows.empty:
                dropped_indices.extend(matching_rows.index.tolist())
            else:
                unmatched_rows.append({
                    'date': date,
                    'islet': islet,
                    'reference': reference,
                    'metadata_subset': filtered_metadata_subset
                })

    # Drop all matching rows from filtered_metadata
    filtered_metadata = filtered_metadata.drop(dropped_indices)

    logging.info(f"Number of rows dropped: {len(dropped_indices)}")
    logging.info(f"Final shape of filtered_metadata: {filtered_metadata.shape}")

    # Log any rows that were not found
    if unmatched_rows:
        logging.info("\nRows not found in filtered_metadata:")
        for row in unmatched_rows:
            logging.info(row)

    return filtered_metadata

def extract_and_encode_label(
    image_name: str
) -> int:

    """Convert string label to numeric :
    alpha --> 0
    beta  --> 1
    """
    if "alpha" in image_name.lower():
        return 0
    elif "beta" in image_name.lower():
        return 1
    else:
        raise ValueError("Label must be 'alpha' or 'beta'.")

def find_contours(
    image: np.ndarray
) -> List[np.ndarray]:
    """Find contours in the binary image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def save_image_df(
    image_df: pd.DataFrame,
    image_df_save_path: str,
    compression_level: int = 3
) -> None:
    """
    Saves the image_df DataFrame using joblib.

    Parameters:
        image_df (pd.DataFrame): The DataFrame to save.
        image_df_save_path (str): Path where the image_df.pkl will be saved.
    """
    # Ensure the directory exists
    save_dir = os.path.dirname(image_df_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Created directory '{save_dir}' for saving image_df.")

    try:
        joblib.dump(image_df, image_df_save_path , compress = compression_level)
        logging.info(f"image_df successfully saved at '{image_df_save_path}'.")
    except Exception as e:
        logging.error(f"Failed to save image_df: {e}")
        raise

def load_and_prepare_metadata(
    csv_path: Path
) -> pd.DataFrame:
    """
    Load metadata, perform initial cleaning, and filter out rows based on rows_to_drop.
    """
    logging.info("Loading metadata...")
    metadata = pd.read_csv(csv_path)

    # Initial cleaning
    metadata['id'] = metadata['name of file'].apply(lambda x: int(x.split('_')[0]))
    metadata['glucose'] = metadata['glucose'].str.replace(' ', '', regex=False).replace('1mM', '16mM', regex=False)
    metadata.rename(columns={'donor': 'date', 'cell type': 'cell_type'}, inplace=True)

    # Ensure correct data types
    metadata['date'] = metadata['date'].astype(str)

    return metadata

def place_on_canvas(
    cropped_image: np.ndarray,
    canvas_size: int = 150
) -> np.ndarray:
    """Pad the cropped image to fit on a 125x125 canvas without resizing. Raise an error if the image is larger."""

    # Get image dimensions
    h, w = cropped_image.shape[:2]

    # Check if the image is larger than the canvas
    if h > canvas_size or w > canvas_size:
        raise ValueError(f"Image size ({w}x{h}) exceeds the canvas size of {canvas_size}x{canvas_size} pixels.")

    # Create a blank canvas
    if len(cropped_image.shape) == 2:  # Grayscale
        padded_canvas = np.zeros((canvas_size, canvas_size), dtype=cropped_image.dtype)
    else:  # Color
        padded_canvas = np.zeros((canvas_size, canvas_size, cropped_image.shape[2]), dtype=cropped_image.dtype)

    # Compute top-left corner to center the image on the canvas
    top_left_y = (canvas_size - h) // 2
    top_left_x = (canvas_size - w) // 2

    # Place the image on the canvas
    padded_canvas[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = cropped_image

    return padded_canvas


def normalize_to_255(image):
    # Convert the image to float for calculation
    image_float = image.astype(np.float64)

    # Find the min and max values of the image
    min_val = np.min(image_float)
    max_val = np.max(image_float)

    # Check if the image is already flat (e.g., all pixels are the same value)
    if min_val == max_val:
        # If flat, set all pixels to a mid-gray value (128)
        normalized_image = np.full(image.shape, 128, dtype=np.uint8)
    else:
        # Normalize the image to the [0, 255] range
        normalized_image = 255 * (image_float - min_val) / (max_val - min_val)
        # Convert to uint8 format
        normalized_image = normalized_image.astype(np.uint8)

    return normalized_image


def process_directory(
    metadata: pd.DataFrame,
    image_dir: str,
    canvas_size: int = 150,
    normalize: bool = False
)-> List[Tuple[np.array, str]]:

    """Process each image in the directory and store transformed images in a list of tuples."""
    processed_images = []  # Initialize an empty list to store the tuples

    # Process each image file in the metadata
    for _, row in metadata.iterrows():
        filename = row['name of file']
        if not filename.endswith('.npy'):
            filename += '.npy'
        file_path = image_dir + "/" + filename

        if os.path.exists(file_path):
            # Load the numpy array (image)
            image = np.load(file_path)

            if normalize:
                # Ensure the image is in uint8 format
                image = normalize_to_255(image)
            else:
                if image.dtype != np.uint8:
                    image = (image*255).astype(np.uint8)

            # Use a low threshold to detect all cells
            binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)[1]

            # Find contours in the image
            contours = find_contours(binary_image)

            # Crop the cells based on contours
            cropped_images = crop_cells(image, contours)

            # If cropped images exist, upscale and pad the first one, and store it in the list
            if cropped_images:
                cropped_canvas_upscaled_padded = place_on_canvas(cropped_images[0], canvas_size = canvas_size)

                # Append the tuple (cropped image, filename) to the list
                processed_images.append((cropped_canvas_upscaled_padded, filename))
        else:
            logging.info(f"Warning: File {file_path} does not exist.")

    return processed_images

def process_and_create_image_df(
    metadata_path: str,
    numpy_images_path: str,
    overlap_threshold: float,
    target_size: int = 150,
    normalize_image:bool = False,
    path_todrop='data/input/Fabio_rowstodrop.csv',
    retention :str = 'balanced'
) -> pd.DataFrame:
    """
    Processes metadata, detects overlaps, processes images, and creates image_df.

    Parameters:
        metadata_path (str): Path to the metadata CSV file.
        numpy_images_path (str): Directory path where .npy image files are stored.
        overlap_threshold (float): Threshold to determine significant overlap.

    Returns:
        pd.DataFrame: DataFrame containing images and labels indexed by filename.
    """
    # Step 2: Load and prepare metadata
    logging.info("Loading and preparing metadata...")
    filtered_metadata = load_and_prepare_metadata(metadata_path)
    logging.info(f"Total files to process: {len(filtered_metadata)}")

    filtered_metadata = drop_rows(filtered_metadata, path_todrop, retention)
    logging.info(f"Remaining files after fabio error analysis: {len(filtered_metadata)}")

    # Step 3: Detect overlapping files
    logging.info("Detecting overlapping files...")
    if overlap_threshold < 1.0:
        overlapping_files = detect_overlaps(filtered_metadata, numpy_images_path, overlap_threshold)
        overlapping_files_cleaned = set(os.path.splitext(file)[0] for file in overlapping_files)
        logging.info("Overlapping cells cleaned.")
        logging.info(f"Total removed files: {len(overlapping_files_cleaned)}")
    else:
        overlapping_files_cleaned = set()


    # Step 4: Prepare non-removed files
    logging.info("Preparing non-removed files...")
    non_removed_files = set(filtered_metadata['name of file']) - overlapping_files_cleaned
    logging.info(f"Total non-removed files: {len(non_removed_files)}")

    # Step 5: Filter metadata to include only non-removed files
    logging.info("Filtering metadata to include only non-removed files...")
    final_metadata = filtered_metadata[filtered_metadata['name of file'].isin(non_removed_files)]
    logging.info(f"Total files to process after removing overlaps: {len(final_metadata)}")

    # Step 6: Process and save images
    logging.info("Processing and saving images...")
    image_data = process_directory(final_metadata, numpy_images_path, canvas_size = target_size, normalize = normalize_image)
    logging.info(f"Processed {len(image_data)} images.")

    # Convert image_data to a DataFrame if it's a list
    if isinstance(image_data, list):
        image_df = pd.DataFrame(image_data, columns=['image', 'filename'])
        logging.info("Converted image_data list to DataFrame.")
    elif isinstance(image_data, pd.DataFrame):
        # Ensure the DataFrame has the required columns
        expected_columns = {'image', 'filename'}
        if not expected_columns.issubset(image_data.columns):
            raise ValueError(f"image_df DataFrame must contain columns: {expected_columns}")
        image_df = image_data.copy()
        logging.info("Verified existing image_data DataFrame.")
    else:
        raise TypeError("image_data must be a list of tuples or a pandas DataFrame.")

    # Set 'filename' as the index for easy access
    image_df.set_index('filename', inplace=True)
    logging.info("Set 'filename' as the index for image_df.")

    # Add 'label' column if not present
    if 'label' not in image_df.columns:
        image_df['label'] = image_df.index.to_series().apply(extract_and_encode_label).astype(int)
        logging.info("Added 'label' column to image_df.")
    else:
        logging.info("'label' column already exists in image_df.")

    return image_df


#  --------------- --------------- --------------- --------------- ---------------
# 1) Load or create the base image_df
#  --------------- --------------- --------------- --------------- ---------------

def load_or_create_image_df(
    image_df_save_path: str,
    numpy_images_path: str,
    metadata_path: str,
    overlap_threshold: int,
    compression_level:int = 3,
    normalize_image: bool = False,
    target_size: int = 150,
    retention: str = 'balanced'
) -> pd.DataFrame:
    """
    Loads an image DataFrame from disk if it exists, or creates it if not.

    Parameters:
        image_df_save_path (str): Path to the saved DataFrame.
        numpy_images_path (str): Path to the numpy image files.
        metadata_path (str): Path to the CSV metadata file.
        overlap_threshold (float): Threshold used for processing images.

    Returns:
        pandas.DataFrame: The loaded or newly created image DataFrame.
    """
    if os.path.exists(image_df_save_path):
        logging.info(f"Loading image DataFrame from '{image_df_save_path}'.")
        try:
            image_df = joblib.load(image_df_save_path)
        except Exception as e:
            logging.error(f"Error loading image DataFrame: {e}")
            image_df = process_and_create_image_df(
                metadata_path = metadata_path,
                numpy_images_path = numpy_images_path,
                overlap_threshold = overlap_threshold,
                normalize_image = normalize_image,
                target_size = target_size,
                retention = retention)
            if image_df is not None:
                save_image_df(image_df, image_df_save_path)
    else:
        logging.info(f"Image DataFrame not found. Creating new image DataFrame at '{image_df_save_path}'.")
        image_df = process_and_create_image_df(
                metadata_path = metadata_path,
                numpy_images_path = numpy_images_path,
                overlap_threshold = overlap_threshold,
                normalize_image = normalize_image,
                target_size = target_size,
                retention = retention)
        if image_df is not None:
            save_image_df(image_df, image_df_save_path, compression_level = compression_level)

    return image_df

