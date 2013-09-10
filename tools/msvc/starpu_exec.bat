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

IF NOT EXIST %STARPUPATH%\AUTHORS GOTO starpunotfound

ECHO.
ECHO Using StarPU in %STARPUPATH%

IF "%1" == "" GOTO invalidparam
IF NOT EXIST %1 GOTO invalidparam

mkdir starpu
FOR %%F IN (%STARPUPATH%\bin\*dll) DO COPY %%F starpu\%%~nF
FOR %%F IN (%STARPUPATH%\bin\*dll) DO COPY %%F starpu
COPY c:\MinGW\bin\pthreadGC2.dll starpu
COPY %STARPUPATH%\lib\libstarpu-1.0.lib starpu

set OLDPATH=%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" x86
echo cd starpu
echo dir %STARPUPATH%\include\starpu\1.0
cl %1 /I%STARPUPATH%\include\starpu\1.0 /link starpu\libstarpu-1.0.lib

set PATH=starpu;%PATH%
.\%~n1.exe

set PATH=%OLDPATH%
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
