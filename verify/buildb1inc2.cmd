@echo off
rem Incremental b1 build (no clean). New file name on purpose so an in-flight build is never
rem overwritten mid-run (cmd reads batch files line by line).
setlocal enabledelayedexpansion
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CUDA_PATH=<CUDA13>
set PATH=%PATH%;<CUDA13>\bin;<CUDA13>\bin\x64;%APPDATA%\Python\Python312;%APPDATA%\Python\Python312\Scripts
set VSLANG=1033
set LOG=<BUILD_ROOT>\b1inc2.log
set NL=<BUILD_ROOT>\b1\.ninja_log
echo === B1INC2 START %DATE% %TIME% === > "%LOG%"
set PREV=-1
set STALL=0
for /L %%K in (1,1,40) do (
  ninja -C <BUILD_ROOT>\b1 -j 8 -k 0 >> "%LOG%" 2>&1
  set RC=!ERRORLEVEL!
  for /f %%C in ('find /c /v "" "%NL%"') do set DONE2=%%C
  echo [iter %%K] ninja rc=!RC! done=!DONE2! %TIME% >> "%LOG%"
  if !RC! EQU 0 ( echo === BUILD OK iter=%%K === >> "%LOG%" & goto :done )
  if !DONE2! LEQ !PREV! ( set /a STALL+=1 ) else ( set STALL=0 )
  set PREV=!DONE2!
  if !STALL! GEQ 4 ( echo === STALLED at done=!DONE2! === >> "%LOG%" & goto :done )
  ping -n 4 127.0.0.1 >nul
)
:done
echo === B1INC2 END %DATE% %TIME% === >> "%LOG%"
