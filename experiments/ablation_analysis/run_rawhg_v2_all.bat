@echo off
REM ============================================================
REM Raw+HG v2 批量运行: 3 datasets x 5 seeds = 15 runs
REM ============================================================
REM 用法: 在 ablation_analysis 目录下双击运行或在 cmd 中执行
REM ============================================================

set PYTHON=python
set SCRIPT=%~dp0run_rawhg_v2.py

echo ============================================================
echo   Raw+HG v2: 3 datasets x 5 seeds
echo ============================================================

for %%D in (OPTC TCE5 DAPT) do (
    for %%S in (2021 2022 2023 2024 2025) do (
        echo.
        echo [%%D] seed=%%S ...
        "%PYTHON%" "%SCRIPT%" --dataset %%D --seed %%S
        if errorlevel 1 (
            echo [ERROR] %%D seed=%%S failed!
        ) else (
            echo [OK] %%D seed=%%S done.
        )
    )
)

echo.
echo ============================================================
echo   All runs completed.
echo   Results in:
echo     progress_OPTC/outputs_OPTC_abl_RawHG_v2/
echo     progress_TCE5/outputs_TCE5_abl_RawHG_v2/
echo     progress_DAPT/outputs_DAPT_abl_RawHG_v2/
echo ============================================================
pause
