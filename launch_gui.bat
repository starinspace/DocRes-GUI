@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title DocRes Studio - Launcher
pushd "%~dp0"
if errorlevel 1 goto LOCATION_ERROR

echo(
echo    DocRes Studio
echo    Starting the desktop workspace...
echo(
if not exist "docres_gui.py" goto FILE_ERROR
if not exist "inference.py" goto FILE_ERROR
if not exist "requirements_gui.txt" goto FILE_ERROR

rem Respect a working Python environment opened by the user.
python -c "import torch, cv2" >nul 2>&1
if not errorlevel 1 goto CHECK_GUI
call :ACTIVATE_ENV
if errorlevel 1 goto ENV_ERROR
python -c "import torch, cv2" >nul 2>&1
if errorlevel 1 goto ENV_ERROR

:CHECK_GUI
python -c "import PySide6.QtWidgets, PIL, numpy" >nul 2>&1
if not errorlevel 1 goto START_GUI
echo    Installing the desktop interface dependencies into this environment...
echo    PyTorch and CUDA are not installed or upgraded by this launcher.
echo(
python -m pip install -r "%~dp0requirements_gui.txt"
if errorlevel 1 goto DEPENDENCY_ERROR
python -c "import PySide6.QtWidgets, PIL, numpy" >nul 2>&1
if errorlevel 1 goto DEPENDENCY_ERROR

:START_GUI
rem Keep the console attached so startup failures stay visible.
rem The model itself runs in a separate process managed by the GUI.
python "%~dp0docres_gui.py"
set "docres_gui_exit=%errorlevel%"
if not "%docres_gui_exit%"=="0" goto GUI_ERROR
popd
exit /b 0

:ACTIVATE_ENV
if /i "%CONDA_DEFAULT_ENV%"=="docresfork" exit /b 0
where conda.bat >nul 2>&1
if not errorlevel 1 goto ACTIVATE_PATH
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" goto ACTIVATE_MINICONDA
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" goto ACTIVATE_ANACONDA
if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" goto ACTIVATE_LOCAL_MINICONDA
exit /b 1
:ACTIVATE_PATH
call conda.bat activate docresfork
exit /b %errorlevel%
:ACTIVATE_MINICONDA
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate docresfork
exit /b %errorlevel%
:ACTIVATE_ANACONDA
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate docresfork
exit /b %errorlevel%
:ACTIVATE_LOCAL_MINICONDA
call "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" activate docresfork
exit /b %errorlevel%

:LOCATION_ERROR
echo    Could not open the folder containing this launcher.
pause
exit /b 1
:FILE_ERROR
echo    Keep docres_gui.py, inference.py, requirements_gui.txt and this BAT
echo    together in your existing DocRes-Fork folder.
goto FAILED
:ENV_ERROR
echo    Could not find a working DocRes Python environment.
echo    Open Anaconda Prompt, run: conda activate docresfork
echo    Then run this BAT again. A working custom environment is also supported.
goto FAILED
:DEPENDENCY_ERROR
echo    The GUI dependencies could not be installed.
echo    Check the error above and your internet connection, then try again.
goto FAILED
:GUI_ERROR
echo    The desktop app closed with an error. See the message above.
goto FAILED
:FAILED
echo(
pause
popd
exit /b 1
