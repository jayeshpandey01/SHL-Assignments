@echo off
echo Audio Grammar Scorer
echo ===================
echo.

if "%~1"=="" (
    echo Usage: run_audio_grammar_scorer.bat [input_file_or_directory] [output_file]
    echo.
    echo Examples:
    echo   run_audio_grammar_scorer.bat dataset/audios_test/audio_706.wav result.csv
    echo   run_audio_grammar_scorer.bat dataset/audios_test results.csv --directory
    echo.
    pause
    exit /b
)

if "%~2"=="" (
    echo Error: Output file not specified
    echo Usage: run_audio_grammar_scorer.bat [input_file_or_directory] [output_file]
    echo.
    pause
    exit /b
)

if "%~3"=="--directory" (
    echo Processing all audio files in directory: %~1
    echo Output will be saved to: %~2
    echo.
    python audio_grammar_scorer.py --input "%~1" --output "%~2" --directory
) else (
    echo Processing audio file: %~1
    echo Output will be saved to: %~2
    echo.
    python audio_grammar_scorer.py --input "%~1" --output "%~2"
)

echo.
echo Processing complete!
pause 