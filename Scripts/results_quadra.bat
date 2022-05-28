@echo off

set BASEDIR=%~dp0\..
set MEMORY=512m
set SINGLE_DRIFT_DIR=D:\Bartek\Studia\Magisterka\Magisterka\Wyniki\QuadraDrift\
set ALGORITHM=\OOB.csv

setlocal enabledelayedexpansion
for %%f in (D:\Bartek\Studia\Magisterka\Magisterka\Dane\QuadraDrift\*) do (
set filename=%%f
set substring=!filename:~56,-5!
set mypath=%SINGLE_DRIFT_DIR%!substring!
set resultpath=!mypath!%ALGORITHM%
mkdir !mypath!
java -Xmx%MEMORY% -cp "%BASEDIR%/lib/*" -javaagent:"%BASEDIR%/lib/sizeofag-1.0.4.jar" moa.DoTask "EvaluatePrequential -l trees.HoeffdingTree -s (ArffFileStream -f %%f) -f 1000 -e (WindowAUCImbalancedPerformanceEvaluator -w 1000) -d !resultpath!")