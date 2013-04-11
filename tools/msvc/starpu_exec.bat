@ECHO OFF

REM StarPU --- Runtime system for heterogeneous multicore architectures.
REM
REM Copyright (C) 2013  Centre National de la Recherche Scientifique
REM
REM StarPU is free software; you can redistribute it and/or modify
REM it under the terms of the GNU Lesser General Public License as published by
REM the Free Software Foundation; either version 2.1 of the License, or (at
REM your option) any later version.
REM
REM StarPU is distributed in the hope that it will be useful, but
REM WITHOUT ANY WARRANTY; without even the implied warranty of
REM MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
REM
REM See the GNU Lesser General Public License in COPYING.LGPL for more details.

TITLE MSVC StarPU Execution
ECHO.
ECHO MSVC StarPU Execution
ECHO.
ECHO %STARPUPATH%

IF "%1" == "" GOTO invalidparam
IF NOT EXIST %1 GOTO invalidparam
IF NOT EXIST %STARPUPATH%\AUTHORS GOTO starpunotfound

COPY %1 starpu\starpu_appli.c
FOR %%F IN (%STARPUPATH%\bin\*dll) DO COPY %%F starpu\%%~nF
COPY c:\MinGW\bin\pthreadGC2.dll starpu
IF EXIST Debug RMDIR /S /Q Debug
IF EXIST starpu\Debug RMDIR /S /Q starpu\Debug

"C:\Program Files (x86)\Microsoft Visual Studio 10.0\Common7\IDE\VCExpress.exe" starpu.sln

GOTO end

:invalidparam
  ECHO.
  ECHO Syntax error. You need to give the name of a StarPU application
  GOTO end

:starpunotfound
  ECHO.
  ECHO You need to set the variable STARPUPATH to a valid StarPU installation directory
  GOTO end

:end
