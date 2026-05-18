@echo off
setlocal enabledelayedexpansion

:: Capture paths passed from Inno Setup
set "GIT_TARGET_DIR=%~1"
set "APP_DIR=%~2"

echo ===================================================
echo STEP 1: Verifying Python Installation...
echo ===================================================
python --version >nul 2>&1
if %errorlevel% neq 0 (
    start "" "https://www.python.org/downloads/"
    echo [!] CRITICAL ERROR: Python is not installed.
    pause
    exit /b 1
)

echo ===================================================
echo STEP 2: Fetching Live Assets From GitHub Repo...
echo ===================================================
echo Downloading repository zip...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $web = New-Object System.Net.WebClient; $web.Headers.Add('User-Agent', 'Mozilla/5.0'); $web.DownloadFile('https://github.com/co230532-a11y/Face-Recognition-with-Streamlit/archive/refs/heads/main.zip', '%TEMP%\repo.zip')"

echo Extracting archive data...
powershell -Command "Expand-Archive -Path '%TEMP%\repo.zip' -DestinationPath '%TEMP%\extracted' -Force"

echo Routing repository modules directly to: "%GIT_TARGET_DIR%"
xcopy /E /Y /I "%TEMP%\extracted\Face-Recognition-with-Streamlit-main\*" "%GIT_TARGET_DIR%" >nul

echo ===================================================
echo STEP 3: Fetching System Shortcut Icon...
echo ===================================================
echo Downloading app_icon.ico from repository assets...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $web = New-Object System.Net.WebClient; $web.Headers.Add('User-Agent', 'Mozilla/5.0'); $web.DownloadFile('https://raw.githubusercontent.com/co230532-a11y/Face-Recognition-with-Streamlit/main/app_icon.ico', '%APP_DIR%\app_icon.ico')" 2>nul

:: FIXED: Removed the accidental "appointments" text typo
if not exist "%APP_DIR%\app_icon.ico" (
    echo [!] Custom icon not found in repository root. Defaulting to system shell icon.
)

echo Cleaning up temporary files...
del "%TEMP%\repo.zip" /f /q
rmdir /s /q "%TEMP%\extracted"

echo ===================================================
echo STEP 4: Initializing Component Packages...
echo ===================================================
python -m pip install --upgrade pip
python -m pip install streamlit
python -m pip install --upgrade --force-reinstall numpy==1.26.4

echo [+] Live system deployment finished flawlessly.
exit /b 0