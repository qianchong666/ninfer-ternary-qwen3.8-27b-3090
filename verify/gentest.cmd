@echo off
rem Reproducible generation probe (ASCII only). Usage: gentest.cmd <label> <artifact> <prompt> <max-new> [extra args...]
setlocal
set LABEL=%~1
set ART=%~2
set PROMPT=%~3
set MAXNEW=%~4
shift & shift & shift & shift
set LOG=<BUILD_ROOT>\gen-%LABEL%.log
set EXE=<BUILD_ROOT>\b1\apps\ninfer.exe
if not exist "%EXE%" ( echo MISSING EXE & exit /b 99 )
if not exist "%ART%" ( echo MISSING ARTIFACT & exit /b 98 )
echo === gen test %LABEL% start %DATE% %TIME% === > "%LOG%"
"%EXE%" "%ART%" --prompt "%PROMPT%" --max-new %MAXNEW% %1 %2 %3 %4 %5 >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo === exit code: %RC% (0x%RC%) === >> "%LOG%"
echo LABEL=%LABEL% EXIT=%RC%
exit /b %RC%
