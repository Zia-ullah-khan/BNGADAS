@echo off
setlocal

:: Define virtual environment directory
set VENV_DIR=venv

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and try again.
    exit /b 1
)

:: Create virtual environment if it does not exist
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

:: Activate the virtual environment
call %VENV_DIR%\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies from requirements.txt
if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error installing dependencies! Trying to install PyTorch separately...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
) else (
    echo requirements.txt not found! Please ensure it exists in the same directory.
    exit /b 1
)

:: Verify numpy installation
python -c "import numpy" 2>nul
if %errorlevel% neq 0 (
    echo Numpy is missing. Reinstalling dependencies...
    pip install --no-cache-dir -r requirements.txt
)

:: Run the Python script
echo Running V0.5ACC.py...
python V0.5ACC.py

:: Keep the window open
pause
