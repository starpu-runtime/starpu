@ECHO OFF
REM StarPU --- Runtime system for heterogeneous multicore architectures.
REM
REM Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
REM
TITLE MSVC StarPU Execution
ECHO.
ECHO MSVC StarPU Execution

IF "%1" == "" GOTO invalidparam
IF NOT EXIST %1 GOTO invalidparam

call .\starpu_var.bat

mkdir starpu
FOR %%F IN (%STARPU_PATH%\bin\*dll) DO COPY %%F starpu\%%~nF
FOR %%F IN (%HWLOC%\bin\*dll) DO COPY %%F starpu

set STARPU_OLDPATH=%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\vcvarsall.bat" x86
cl %1 %STARPU_CFLAGS% %STARPU_LDFLAGS%

set PATH=starpu;c:\MinGW\bin;%PATH%
.\%~n1.exe

set PATH=%STARPU_OLDPATH%
GOTO end

:invalidparam
  ECHO.
  ECHO Syntax error. You need to give the name of a StarPU application
  EXIT /B 2
  GOTO end

:end
