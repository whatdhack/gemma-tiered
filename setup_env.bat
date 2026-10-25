@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo [1/4] Checking Conda / Miniforge Installation...
echo =======================================================

where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda not found in PATH. Checking common Miniforge paths...
    if exist "%USERPROFILE%\miniforge3\condabin\conda.bat" (
        set "CONDA_CMD=%USERPROFILE%\miniforge3\condabin\conda.bat"
    ) else if exist "C:\miniforge3\condabin\conda.bat" (
        set "CONDA_CMD=C:\miniforge3\condabin\conda.bat"
    ) else (
        echo Miniforge not found. Installing Miniforge3 via winget...
        winget install -e --id CondaForge.Miniforge3 --accept-source-agreements --accept-package-agreements
        if exist "%USERPROFILE%\miniforge3\condabin\conda.bat" (
            set "CONDA_CMD=%USERPROFILE%\miniforge3\condabin\conda.bat"
        ) else (
            echo [ERROR] Installation finished but conda.bat not located. Please restart shell.
            pause
            exit /b 1
        )
    )
) else (
    set "CONDA_CMD=conda"
)

echo Using Conda command: %CONDA_CMD%

echo =======================================================
echo [2/4] Registering C:\aiw\envs directory with Conda...
echo =======================================================
if not exist "C:\aiw\envs" (
    mkdir "C:\aiw\envs"
)
call %CONDA_CMD% config --add envs_dirs C:\aiw\envs

echo =======================================================
echo [3/4] Creating Python 3.12 environment in C:\aiw\envs\gemma-tiered...
echo =======================================================
call %CONDA_CMD% create --prefix C:\aiw\envs\gemma-tiered python=3.12 -y

echo =======================================================
echo [4/4] Installing Required Packages...
echo =======================================================
set "ENV_PIP=C:\aiw\envs\gemma-tiered\Scripts\pip.exe"
if not exist "%ENV_PIP%" (
    echo [ERROR] Could not find pip at %ENV_PIP%.
    pause
    exit /b 1
)

echo Installing dependencies via: %ENV_PIP%
"%ENV_PIP%" install requests pydantic rich

echo Attempting to install LiteRT packages (Note: if wheels are unavailable on PyPI for 3.12, router will use fallback reflex classifier)...
"%ENV_PIP%" install ai-edge-litert litert-lm-api || echo [INFO] Proceeding with high-speed local classifier fallback if LiteRT is not yet available for Python 3.12 on Windows.

echo =======================================================
echo Setup complete!
echo To activate:
echo   conda activate C:\aiw\envs\gemma-tiered
echo   (or simply: conda activate gemma-tiered)
echo =======================================================
pause
