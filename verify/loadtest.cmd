@echo off
rem Reproducible artifact-load probe (ASCII only, per env pitfall #1/#5).
rem Usage: loadtest.cmd <label> <artifact.ninfer>
rem Writes <BUILD_ROOT>\load-<label>.log and prints the exit code.
setlocal
set LABEL=%~1
set ART=%~2
set LOG=<BUILD_ROOT>\load-%LABEL%.log
set EXE=<BUILD_ROOT>\b1\apps\ninfer.exe

if not exist "%EXE%" ( echo MISSING EXE %EXE% & exit /b 99 )
if not exist "%ART%" ( echo MISSING ARTIFACT %ART% & exit /b 98 )

echo === load test %LABEL% start %DATE% %TIME% === > "%LOG%"
echo artifact: %ART% >> "%LOG%"
echo exe: %EXE% >> "%LOG%"
"%EXE%" "%ART%" --prompt hi --max-new 1 >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo === exit code: %RC% (0x%RC%) === >> "%LOG%"
echo LABEL=%LABEL% EXIT=%RC%
exit /b %RC%
