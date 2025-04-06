@echo off
echo Setting up directories...
python setup_directories.py

echo.
echo Running batch grammar assistant...
python batch_grammar_assistant.py

echo.
echo Processing complete!
pause 