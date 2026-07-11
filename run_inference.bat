@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo  Interactive Tagger Batch Inference Runner
echo ====================================================
echo.

:GET_FOLDER
set /p folder_path="Enter the path to the folder containing images: "
if "%folder_path%"=="" (
    echo Error: Folder path cannot be empty.
    goto GET_FOLDER
)

rem Strip quotes if the user dragged and dropped the folder
set "folder_path=%folder_path:"=%"

if not exist "%folder_path%" (
    echo Error: The folder "%folder_path%" does not exist.
    goto GET_FOLDER
)

set /p model="Enter Hugging Face model repo ID or local path [Default: Mooshie/caformer_b36.dbv4-full]: "
if "%model%"=="" (
    set "model=Mooshie/caformer_b36.dbv4-full"
)

set /p gen_th="Enter threshold for general tags [Default: 0.39]: "
if "%gen_th%"=="" (
    set "gen_th=0.39"
)

set /p char_th="Enter threshold for character tags [Default: 0.47]: "
if "%char_th%"=="" (
    set "char_th=0.47"
)

set /p rating_th="Enter threshold for rating tags [Default: 0.39]: "
if "%rating_th%"=="" (
    set "rating_th=0.39"
)

set /p batch_size="Enter batch size to prevent VRAM OOM [Default: 16]: "
if "%batch_size%"=="" (
    set "batch_size=16"
)

set /p hf_token="Enter Hugging Face token (press Enter if none) [Default: none]: "

echo.
echo Starting Inference with:
echo   Folder: %folder_path%
echo   Model: %model%
echo   General Threshold: %gen_th%
echo   Character Threshold: %char_th%
echo   Rating Threshold: %rating_th%
echo   Batch Size: %batch_size%
if not "%hf_token%"=="" (
    echo   HF Token: [HIDDEN]
)
echo.

if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment 'venv' not found in current directory.
    goto END
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Running batch inference...
if "%hf_token%"=="" (
    python batch_inference.py "%folder_path%" --model "%model%" --gen-threshold %gen_th% --char-threshold %char_th% --rating-threshold %rating_th% --batch-size %batch_size%
) else (
    python batch_inference.py "%folder_path%" --model "%model%" --hf-token "%hf_token%" --gen-threshold %gen_th% --char-threshold %char_th% --rating-threshold %rating_th% --batch-size %batch_size%
)

:END
echo.
pause
