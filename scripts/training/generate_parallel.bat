@echo off
REM Parallel dataset generation script for Windows
REM Usage: generate_parallel.bat <endgame> [workers] [chunk_size]
REM Example: generate_parallel.bat KRRvK 8 10000

setlocal

if "%1"=="" (
    echo Usage: generate_parallel.bat ^<endgame^> [workers] [chunk_size]
    echo.
    echo Examples:
    echo   generate_parallel.bat KQvK
    echo   generate_parallel.bat KRRvK 8 10000
    echo   generate_parallel.bat KRvKP 6 5000
    echo.
    echo Available endgames:
    echo   3-piece: KQvK, KRvK, KPvK, KBvK, KNvK
    echo   4-piece: KRRvK, KRvKP, KPvKP, KBPvK, KQvKR, etc.
    exit /b 1
)

set ENDGAME=%1
set WORKERS=%2
set CHUNK_SIZE=%3

if "%WORKERS%"=="" set WORKERS=8
if "%CHUNK_SIZE%"=="" set CHUNK_SIZE=10000

echo ============================================================
echo PARALLEL DATASET GENERATION
echo ============================================================
echo Endgame: %ENDGAME%
echo Workers: %WORKERS%
echo Chunk size: %CHUNK_SIZE%
echo ============================================================
echo.

python src/generate_datasets_parallel.py ^
    --config %ENDGAME% ^
    --relative ^
    --workers %WORKERS% ^
    --chunk-size %CHUNK_SIZE%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS: Dataset generated successfully!
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo ERROR: Dataset generation failed!
    echo ============================================================
    exit /b 1
)

endlocal
