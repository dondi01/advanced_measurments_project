@echo off
REM Run TensorFlow GPU container with all required data folders mounted

setlocal

REM Set image name
set IMAGE=test4

REM Set project root (current directory)
set PROJECT_ROOT=%cd%

REM Run the container
docker run --gpus all -it --rm ^
  -v "%PROJECT_ROOT%:/app" ^
  -v "%PROJECT_ROOT%\data:/app/data" ^
  -v "%PROJECT_ROOT%\dataset_medi:/app/dataset_medi" ^
  -v "%PROJECT_ROOT%\dataset_piccoli:/app/dataset_piccoli" ^
  -v "%PROJECT_ROOT%\ml_datasets:/app/ml_datasets" ^
  -v "%PROJECT_ROOT%\Old_Reconstructions:/app/Old_Reconstructions" ^
  -v "%PROJECT_ROOT%\Reconstructed:/app/Reconstructed" ^
  -v "%PROJECT_ROOT%\Schematics:/app/Schematics" ^
  -w /app ^
  %IMAGE% bash

endlocal