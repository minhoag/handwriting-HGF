# 1. Download only if missing
if (-not (Test-Path "files-archive.zip")) {
    Write-Host "Downloading Zenodo archive..." -ForegroundColor Cyan
    wget "https://zenodo.org/api/records/56198/files-archive" -OutFile "files-archive.zip" -UseBasicParsing
}

# 2. Extract main archive
Write-Host "Extracting files..." -ForegroundColor Cyan
Expand-Archive -Path "files-archive.zip" -DestinationPath "." -Force

# 3. Extract the formula images tarball
if (Test-Path "formula_images.tar.gz") {
    tar -xvzf "formula_images.tar.gz"
}

# 4. Handle the inkML data (from your other file: archive.zip)
if (Test-Path "archive.zip") {
    Expand-Archive -Path "archive.zip" -DestinationPath "inkML_data" -Force
}

# 5. Fix the folder structure (Flattening)
Write-Host "Organizing folders..." -ForegroundColor Cyan
Move-Item "inkML_data\TrainINKML_2013\TrainINKML\*" "inkML_data\TrainINKML_2013\" -ErrorAction SilentlyContinue
Move-Item "inkML_data\trainData_2012_part1\trainData_2012_part1\*" "inkML_data\trainData_2012_part1\" -ErrorAction SilentlyContinue
Move-Item "inkML_data\trainData_2012_part2\trainData_2012_part2\*" "inkML_data\trainData_2012_part2\" -ErrorAction SilentlyContinue

# 6. Ensure the 2011 folder exists so the Python script doesn't crash
if (-not (Test-Path "inkML_data\CROHME_training_2011")) { 
    New-Item -ItemType Directory -Path "inkML_data\CROHME_training_2011" 
}

# 7. Run the Python Preprocessor
Write-Host "Starting Preprocessing..." -ForegroundColor Green
python run_preprocess.py