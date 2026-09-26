@echo off
echo Installing PyInstaller...
pip install pyinstaller

echo.
echo Building Executable...
REM We use --onedir instead of --onefile because PyTorch and XGBoost are massive.
REM A single .exe file would take minutes just to extract itself every time you run it.
pyinstaller --noconfirm --onedir --name EntityResolutionPipeline src/inference.py

echo.
echo Build Complete!
echo You can find your executable inside the "dist\EntityResolutionPipeline" folder.
echo You can zip that entire folder and share it with others.
pause
