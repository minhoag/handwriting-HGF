#!/bin/bash

# Set target directory relative to the repository root
TARGET_DIR="inkML_data"

# Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Downloading CROHME 2019 dataset from Kaggle..."
# Download the dataset directly into the target directory
# Note: This URL may require Kaggle authentication if not using a pre-signed link
curl -L -o "$TARGET_DIR/crohme2019.zip" \
  https://www.kaggle.com/api/v1/datasets/download/ntcuong2103/crohme2019

if [ $? -eq 0 ]; then
    echo "Download successful. Unzipping..."
    # Unzip the contents into the inkML_data folder
    # -o overwrites existing files to perform an "update"
    unzip -o "$TARGET_DIR/crohme2019.zip" -d "$TARGET_DIR"
    
    echo "Cleaning up zip file..."
    rm "$TARGET_DIR/crohme2019.zip"
    
    echo "Dataset update complete. Files are in: $TARGET_DIR"
    echo "You can now run 'python run_preprocess.py' to process the new data."
else
    echo "Error: Download failed. Please ensure the Kaggle URL is valid and accessible."
    exit 1
fi
