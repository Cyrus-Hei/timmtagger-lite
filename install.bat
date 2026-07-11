@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo  Tagger Virtual Environment and Dependency Installer
echo ====================================================
echo.

rem Check for Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python (recommended version 3.10 or 3.11) and try again.
    goto END
)

rem Get Python version
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set py_ver=%%v
echo Found Python version: %py_ver%

rem Check if virtual environment already exists
if exist "venv\Scripts\activate.bat" (
    echo Virtual environment 'venv' already exists.
) else (
    echo Creating virtual environment 'venv'...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo Error: Failed to create virtual environment.
        goto END
    )
    echo Virtual environment created successfully.
)

echo.
echo Activating virtual environment and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo Installing project dependencies from requirements.txt...
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo Error: Failed to install project dependencies.
    goto END
)
echo Dependencies installed successfully.
echo.

:ASK_DOWNLOAD
set /p dl_choice="Do you want to pre-download a model for offline usage? (Y/N) [Default: N]: "
if "%dl_choice%"=="" set "dl_choice=N"

if /i "%dl_choice%"=="Y" (
    echo.
    set /p model_repo="Enter Hugging Face repository ID [Default: Mooshie/caformer_b36.dbv4-full]: "
    if "!model_repo!"=="" set "model_repo=Mooshie/caformer_b36.dbv4-full"

    set /p local_folder="Enter local folder path to save model [Default: ./local_model]: "
    if "!local_folder!"=="" set "local_folder=./local_model"

    set /p hf_token="Enter Hugging Face token (press Enter if none) [Default: none]: "

    echo.
    if "!hf_token!"=="" (
        python download_model.py --repo "!model_repo!" --dir "!local_folder!"
    ) else (
        python download_model.py --repo "!model_repo!" --dir "!local_folder!" --token "!hf_token!"
    )
    if !errorlevel! neq 0 (
        echo Warning: Model download failed. You can re-run this install or run online later.
    ) else (
        echo.
        echo Model successfully downloaded and saved to: !local_folder!
    )
)

echo.
echo ====================================================
echo  Installation and Setup complete!
echo  You can now run 'run_inference.bat' to tag images.
echo ====================================================
echo.

:END
pause
