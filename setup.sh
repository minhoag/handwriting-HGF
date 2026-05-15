#!/usr/bin/env bash
set -euo pipefail

# 1. Download only if missing
if [[ ! -f "files-archive.zip" ]]; then
  echo "Downloading Zenodo archive..."
  curl -L "https://zenodo.org/api/records/56198/files-archive" -o "files-archive.zip"
fi

# 2. Extract main archive
echo "Extracting files..."
unzip -o "files-archive.zip"

# 3. Extract the formula images tarball
if [[ -f "formula_images.tar.gz" ]]; then
  tar -xvzf "formula_images.tar.gz"
fi

# 4. Handle the handwritten inkML data
mkdir -p "inkML_data/TrainINKML_2013"
mkdir -p "inkML_data/trainData_2012_part1"
mkdir -p "inkML_data/trainData_2012_part2"
mkdir -p "inkML_data/CROHME_training_2011"

# Use local handwritten archive if present. Otherwise download it.
handwritten_archive="handwritten-mathematical-expressions.zip"

if [[ -f "archive.zip" ]]; then
  handwritten_archive="archive.zip"
elif [[ ! -f "$handwritten_archive" ]]; then
  echo "Downloading handwritten inkML dataset from Kaggle..."
  echo "This requires Kaggle API access/authentication to succeed."
  curl -L \
    -o "$handwritten_archive" \
    "https://www.kaggle.com/api/v1/datasets/download/rtatman/handwritten-mathematical-expressions"
fi

unzip -o "$handwritten_archive" -d "inkML_data"

# 5. Fix the folder structure (Flattening)
echo "Organizing folders..."
if [[ -d "inkML_data/TrainINKML_2013/TrainINKML" ]]; then
  mv inkML_data/TrainINKML_2013/TrainINKML/* inkML_data/TrainINKML_2013/ 2>/dev/null || true
fi
if [[ -d "inkML_data/trainData_2012_part1/trainData_2012_part1" ]]; then
  mv inkML_data/trainData_2012_part1/trainData_2012_part1/* inkML_data/trainData_2012_part1/ 2>/dev/null || true
fi
if [[ -d "inkML_data/trainData_2012_part2/trainData_2012_part2" ]]; then
  mv inkML_data/trainData_2012_part2/trainData_2012_part2/* inkML_data/trainData_2012_part2/ 2>/dev/null || true
fi

# 6. Run the Python Preprocessor
echo "Starting Preprocessing..."
python run_preprocess.py
