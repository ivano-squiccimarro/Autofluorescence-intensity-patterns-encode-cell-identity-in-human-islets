import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
from pathlib import Path

# Updated imports to point to the new subfolders
from src.pipeline_seed import pipeline_seed
from config.pipeline_configuration import PipelineConfig
from src.from_images_collection_to_dataframe import load_or_create_image_df

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


    base_results = Path("Dataset++") 
    base_results.mkdir(exist_ok=True)

    # 2) Override the image_df path to live inside the results folder
    image_df_path = base_results / "image_df.pkl"

    # 3) Load the configuration
    config = PipelineConfig()

    # 4) Build or load the image_df and save it
    image_df = load_or_create_image_df(
        image_df_save_path = str(image_df_path),
        numpy_images_path  = str(config.numpy_images_path),
        metadata_path      = str(config.metadata_path),
        overlap_threshold  = config.overlap_threshold,
        compression_level  = config.compression_level,
        target_size        = config.width,
        normalize_image    = config.image_normalize,
        retention          = 'balanced'
    )
    # 5) Run the pipeline for all seeds defined in the config
    for seed in config.random_seeds:
        logging.info(f"=== Running seed {seed} ===")
        pipeline_seed(
            gen_results_folder = str(base_results),
            seed               = seed,
            image_df           = image_df,
            config             = config
        )

if __name__ == "__main__":
    main()