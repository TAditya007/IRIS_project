@echo off
REM ============================================================
REM  create_project.bat
REM  Builds the iris_replication/ folder tree. Idempotent.
REM  Save as: UTF-8 (no BOM), line endings: CRLF
REM ============================================================
setlocal

echo.
echo ============================================================
echo   iRIS Replication - Project Scaffolder
echo ============================================================
echo.

REM --- Resolve project root (parent of this script's folder) ---
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%") do set "SETUP_DIR=%%~fI"
for %%I in ("%SETUP_DIR%\..") do set "ROOT=%%~fI"
set "PROJ=%ROOT%\iris_replication"

echo [INFO] Setup folder : %SETUP_DIR%
echo [INFO] Project root : %ROOT%
echo [INFO] Project dir  : %PROJ%
echo.

REM --- Create project root ---
if not exist "%PROJ%" (
    mkdir "%PROJ%"
    echo [OK]   Created: iris_replication\
) else (
    echo [SKIP] Exists : iris_replication\
)

REM --- Subfolders ---
if not exist "%PROJ%\configs"          mkdir "%PROJ%\configs"
if not exist "%PROJ%\environment"      mkdir "%PROJ%\environment"
if not exist "%PROJ%\agent"            mkdir "%PROJ%\agent"
if not exist "%PROJ%\analysis"         mkdir "%PROJ%\analysis"
if not exist "%PROJ%\checkpoints"      mkdir "%PROJ%\checkpoints"
if not exist "%PROJ%\logs"             mkdir "%PROJ%\logs"
if not exist "%PROJ%\results"          mkdir "%PROJ%\results"
if not exist "%PROJ%\results\figures"  mkdir "%PROJ%\results\figures"
if not exist "%PROJ%\data"             mkdir "%PROJ%\data"
echo [OK]   Subfolders ensured.

REM --- Placeholder files ---
if not exist "%PROJ%\configs\default.yaml"           type nul > "%PROJ%\configs\default.yaml"
if not exist "%PROJ%\environment\__init__.py"        type nul > "%PROJ%\environment\__init__.py"
if not exist "%PROJ%\environment\channel.py"         type nul > "%PROJ%\environment\channel.py"
if not exist "%PROJ%\environment\allocation.py"      type nul > "%PROJ%\environment\allocation.py"
if not exist "%PROJ%\environment\ris_env.py"         type nul > "%PROJ%\environment\ris_env.py"
if not exist "%PROJ%\environment\quantization.py"    type nul > "%PROJ%\environment\quantization.py"
if not exist "%PROJ%\agent\__init__.py"              type nul > "%PROJ%\agent\__init__.py"
if not exist "%PROJ%\agent\replay_buffer.py"         type nul > "%PROJ%\agent\replay_buffer.py"
if not exist "%PROJ%\agent\dqn.py"                   type nul > "%PROJ%\agent\dqn.py"
if not exist "%PROJ%\analysis\__init__.py"           type nul > "%PROJ%\analysis\__init__.py"
if not exist "%PROJ%\analysis\says_fit.py"           type nul > "%PROJ%\analysis\says_fit.py"
if not exist "%PROJ%\analysis\quantization.py"       type nul > "%PROJ%\analysis\quantization.py"
if not exist "%PROJ%\analysis\fairness.py"           type nul > "%PROJ%\analysis\fairness.py"
if not exist "%PROJ%\train.py"                       type nul > "%PROJ%\train.py"
if not exist "%PROJ%\evaluate.py"                    type nul > "%PROJ%\evaluate.py"
if not exist "%PROJ%\requirements.txt"               type nul > "%PROJ%\requirements.txt"
if not exist "%PROJ%\README.md"                      type nul > "%PROJ%\README.md"
if not exist "%PROJ%\commands.txt"                   type nul > "%PROJ%\commands.txt"
if not exist "%PROJ%\timeline.txt"                   type nul > "%PROJ%\timeline.txt"
if not exist "%PROJ%\work done.txt"                  type nul > "%PROJ%\work done.txt"
echo [OK]   Placeholder files ensured.

REM --- Seed tracker headers (only when file size is 0) ---
call :seed "%PROJ%\commands.txt"  "iRIS REPLICATION -- COMMAND LOG"     "Every shell command executed is appended here."
call :seed "%PROJ%\timeline.txt"  "iRIS REPLICATION -- TIMELINE"        "Format: [START] - [END] | PHASE | STATUS"
call :seed "%PROJ%\work done.txt" "iRIS REPLICATION -- WORK DONE LOG"   "Append-only log. One section per phase."

echo.
echo ============================================================
echo   Final tree:
echo ============================================================
pushd "%PROJ%"
tree /F
popd

echo.
echo [DONE] Scaffold ready at:
echo        %PROJ%
echo.
pause
endlocal
exit /b 0


REM ============================================================
REM  :seed <file> <title> <subtitle>
REM  Writes a 4-line header into <file> if and only if it is 0 bytes.
REM ============================================================
:seed
set "SEED_FILE=%~1"
set "SEED_TITLE=%~2"
set "SEED_SUB=%~3"
for %%A in ("%SEED_FILE%") do set "SEED_SIZE=%%~zA"
if "%SEED_SIZE%"=="0" (
    echo ======================================== >  "%SEED_FILE%"
    echo %SEED_TITLE%                           >> "%SEED_FILE%"
    echo ======================================== >> "%SEED_FILE%"
    echo %SEED_SUB%                             >> "%SEED_FILE%"
    echo.                                       >> "%SEED_FILE%"
    echo [OK]   Seeded: %~nx1
) else (
    echo [SKIP] Non-empty: %~nx1
)
exit /b 0