REM StarPU --- Runtime system for heterogeneous multicore architectures.
REM
REM Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

set PATH=%PATH%;C:\MinGW\msys\1.0\bin;c:\msys64\usr\bin;c:\Program Files (x86)\Microsoft Visual Studio 11.0\Common7\IDE;c:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin;C:\Users\Administrator\AppData\Local\Programs\Python\Python37-32
sh -c "./job-1-build-windows.sh"
set PATH=C:\Windows\SysWOW64;C:\Program Files (x86)\Mozilla Firefox;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\SysWOW64;C:\Program Files\Java\jre7\bin;
set HWLOC=c:\StarPU\hwloc-win32-build-1.11.0

cd starpu_install
set STARPU_PATH=%cd%
cd bin\msvc
starpu_exec ../../share/doc/starpu/tutorial/hello_world_msvc.c

