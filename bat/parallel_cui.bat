@echo off
set /P N="How many loops do you want?"
for /L %%i in (1,1,%N%) do (
start %1
)
