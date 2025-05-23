<!---
 StarPU --- Runtime system for heterogeneous multicore architectures.

 Copyright (C) 2009-2025    University of Bordeaux, CNRS (LaBRI UMR 5800), Inria

 StarPU is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or (at
 your option) any later version.

 StarPU is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

 See the GNU Lesser General Public License in COPYING.LGPL for more details.
-->

# Installing StarPU

## Installing StarPU on a Unix machine

```shell
$ ./autogen.sh # If running the SVN version
$ ./configure --prefix=<prefix>
$ make
$ make install
```

## Installing StarPU on Windows

If you are building from a tarball downloaded from the website, you can skip the
cygwin part.

### Install cygwin

http://cygwin.com/install.html

Make sure the following packages are available:
- (Devel)/subversion
- (Devel)/libtool
- (Devel)/gcc
- (Devel)/make
- your favorite editor (vi, emacs, ...)
- (Devel)/gdb
- (Archive)/zip
- (Devel)/pkg-config

### Install mingw

http://www.mingw.org/

### Install hwloc (not mandatory, but strongly recommended)

http://www.open-mpi.org/projects/hwloc

Be careful which version you are installing. Even if your machine
runs windows 64 bits, if you are running a 32 bits mingw (check the
output of the command `uname -a`), you will need to install the 32
bits version of hwloc.

### Install Microsoft Visual C++ Studio Express

http://www.microsoft.com/express/Downloads

Add in your path the following directories.
(adjusting where necessary for the Installation location according to VC
 version and on 64 and 32bit Windows versions)

On cygwin, with Visual C++ 2010 e.g.;

```shell
export PATH="/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 10.0/Common7/IDE":$PATH
export PATH="/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 10.0/VC/bin":$PATH
```

On MingW, with Visual C++ 2010, e.g.;

```shell
export PATH="/c/Program Files (x86)/Microsoft Visual Studio 10.0/Common7/IDE":$PATH
export PATH="/c/Program Files (x86)/Microsoft Visual Studio 10.0/VC/bin":$PATH
```

Try to call `cl.exe`, `lib.exe` and `link.exe` without any option to make
sure these dump their help output with a series of options, otherwise no
`.def` or `.lib` file will be produced.

### Install GPU Drivers (not mandatory)

#### Install Cuda

http://developer.nvidia.com/object/cuda_3_2_downloads.html

You need to install at least the CUDA toolkit.

`libtool` is not able to find the libraries automatically, you
need to make some copies:

```shell
copy c:\cuda\lib\cuda.lib c:\cuda\lib\libcuda.lib
copy c:\cuda\lib\cudart.lib c:\cuda\lib\libcudart.lib
copy c:\cuda\lib\cublas.lib c:\cuda\lib\libcublas.lib
copy c:\cuda\lib\cufft.lib c:\cuda\lib\libcufft.lib
copy c:\cuda\lib\OpenCL.lib c:\cuda\lib\libOpenCL.lib
```

(and if the version of your CUDA driver is >= 3.2)

```shell
copy c:\cuda\lib\curand.lib c:\cuda\lib\libcurand.lib
```

Add the CUDA bin directory in your path

```shell
export PATH=/cygdrive/c/CUDA/bin:$PATH
```

Since we build code using CUDA headers with gcc instead of Visual studio,
a fix is needed: `c:\cuda\include\host_defines.h` has a bogus `CUDARTAPI`
definition which makes linking fail completely. Replace the first
occurrence of

```C
#define CUDARTAPI
```

with

```C
#ifdef _WIN32
#define CUDARTAPI __stdcall
#else
#define CUDARTAPI
#endif
```

While at it, you can also comment the `__cdecl` definition to avoid spurious
warnings.

#### Install OpenCL

http://developer.nvidia.com/object/opencl-download.html

You need to download the NVIDIA Drivers for your version of
Windows. Executing the file will extract all files in a given
directory. The the driver installation will start, it will fail
if no compatibles drivers can be found on your system.

Anyway, you should copy the `*.dl_` files from the directory
(extraction path) in the bin directory of the CUDA installation
directory (the directory should be `v3.2/bin/`)

#### Install MsCompress

http://gnuwin32.sourceforge.net/packages/mscompress.htm

Go in the CUDA bin directory, uncompress `.dl_` files and rename
them in `.dll` files

```shell
cp /cygdrive/c/NVIDIA/DisplayDriver/190.89/International/*.dl_ .
for i in *.dl_ ; do /cygdrive/c/Program\ Files/GnuWin32/bin/msexpand.exe  $i ; mv ${i%_} ${i%_}l ; done
```

If you are building from a tarball downloaded from the website, you can skip the
`autogen.sh` part.

### Start autogen.sh from cygwin

```shell
cd starpu-trunk
./autogen.sh
```

### Start a MinGW shell

```shell
/cygdrive/c/MinGW/msys/1.0/bin/sh.exe --login -i
```

### Configure, make, install from MinGW

If you have a non-english version of windows, use

```shell
export LANG=C
```

otherwise `libtool` has troubles parsing the translated output of the toolchain.

```
cd starpu
mkdir build
cd build
../configure --prefix=$PWD/target \
     --with-hwloc=<HWLOC installation directory> \
     --with-cuda-dir=<CUDA installation directory> \
     --with-cuda-lib-dir=<CUDA installation directory>/lib/Win32 \
	--with-opencl-dir=<CUDA installation directory>
     --disable-build-doc
	--disable-build-examples --enable-quick-check
make
make check   # not necessary but well advised
make install
```

To fasten the compilation process, the option
`--disable-build-examples` may be used to disable the
compilation of the applications in the examples directory. Only the
applications in the test directory will be built.

Also convert a couple of files to CRLF:

```shell
sed -e 's/$/'$'\015'/ < README > $prefix/README.txt
sed -e 's/$/'$'\015'/ < AUTHORS > $prefix/AUTHORS.txt
sed -e 's/$/'$'\015'/ < COPYING.LGPL > $prefix/COPYING.LGPL.txt
```

### Standalone installation

If you want your StarPU installation to be standalone, you need to
copy the DLL files from hwloc, Cuda, and OpenCL into the StarPU
installation bin directory, as well as `MinGW/bin/libpthread*dll`

```shell
cp <CUDA directory>/bin/*dll target/bin
cp <HWLOC directory>/bin/*dll target/bin
cp /cygdrive/c/MinGW/bin/libpthread*dll target/bin
```

and set the StarPU bin directory in your path.

```shell
export PATH=<StarPU installation directory>/bin:$PATH
```
