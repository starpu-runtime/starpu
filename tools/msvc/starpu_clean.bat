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
TITLE MSVC StarPU Cleaning
ECHO.
ECHO MSVC StarPU Cleaning
ECHO.

FOR %%d in (debug starpu\debug ipch) DO IF EXIST %%d RMDIR /S /Q %%d
FOR %%f in (starpu.sdf starpu.suo) DO IF EXIST %%f DEL %%f

