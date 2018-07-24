@echo off
::To evaluate each iterations

set logDir=D:\julien\project\3D-Unet\results\prediction
set modelDir=D:\julien\project\3D-Unet\results\training
set PYTHON=C:\Users\simizlab\Anaconda3\python.exe
set runPy=D:\julien\project\3D-Unet\predict.py

call :run	500
call :run	1000
call :run	1500
call :run	2000
call :run	2500
call :run	3000
call :run	3500
call :run	4000
call :run	4000
call :run	5500
call :run	5500
call :run	6000
call :run	6500
call :run	7000
call :run	7500
call :run	8000
call :run	8500
call :run	9000
call :run	9500
call :run	10000
call :run	10500
call :run	11000
call :run	11500
call :run	12000
call :run	12500
call :run	13000
call :run	13500
call :run	14000
call :run	14500
call :run	15000



PAUSE
exit

:run
set arg1=%1
mkdir %logDir%\%arg1%
if ERRORLEVEL 1 GOTO :skip

echo;
echo MakeDirectory %logDir%\%arg1%

::Analysis
PYTHON %runPy%  -g 0 -m %modelDir%\UNet3D_%arg1%.npz -o %logDir%\%arg1%

:skip
exit /b
