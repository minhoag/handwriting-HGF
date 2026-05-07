$TargetDir = "inkML_data"
$Url = "https://www.kaggle.com/api/v1/datasets/download/ntcuong2103/crohme2019"
$ZipFile = Join-Path $TargetDir "crohme2019.zip"

# Create directory if it doesn't exist
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir | Out-Null
    Write-Host "Created directory: $TargetDir"
}

Write-Host "Downloading CROHME 2019 dataset from Kaggle..."
try {
    # Using Invoke-WebRequest to download the file
    Invoke-WebRequest -Uri $Url -OutFile $ZipFile
    
    Write-Host "Download successful. Unzipping..."
    # -Force allows overwriting existing files for an "update"
    Expand-Archive -Path $ZipFile -DestinationPath $TargetDir -Force
    
    Write-Host "Cleaning up zip file..."
    Remove-Item -Path $ZipFile
    
    Write-Host "Dataset update complete. Files are in: $TargetDir"
    Write-Host "You can now run 'python run_preprocess.py' to process the new data."
}
catch {
    Write-Host "Error: Failed to download or process dataset. Check your internet connection or Kaggle URL." -ForegroundColor Red
    exit 1
}
