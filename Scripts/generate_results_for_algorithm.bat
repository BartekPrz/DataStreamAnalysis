@echo off

set BASEDIR=%~dp0\..
set MEMORY=512m
set RESULTS_DIR=%~dp0\..\Results\
set ALGORITHM=\OOB.csv

setlocal enabledelayedexpansion
for %%f in (%~dp0\..\DataStreams\*) do (
set filename=%%f
set substring=!filename:~56,-5!
set mypath=%RESULTS_DIR%!substring!
set resultpath=!mypath!%ALGORITHM%
mkdir !mypath!
java -Xmx%MEMORY% -cp "%BASEDIR%/lib/*" -javaagent:"%BASEDIR%/lib/sizeofag-1.0.4.jar" moa.DoTask "EvaluatePrequential -l (meta.OOB -s 15) -s (ArffFileStream -f %%f) -f 1000 -e (WindowAUCImbalancedPerformanceEvaluator -w 1000) -d !resultpath!")