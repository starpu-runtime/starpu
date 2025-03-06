<!---
StarPU --- Runtime system for heterogeneous multicore architectures.

Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria

StarPU is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at
your option) any later version.

StarPU is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU Lesser General Public License in COPYING.LGPL for more details.
-->

# StarPU: A Unified Runtime System for Heterogeneous Multicore Architectures

## What is StarPU?

StarPU is a runtime system that offers support for heterogeneous multicore
machines. While many efforts are devoted to design efficient computation kernels
for those architectures (e.g. to implement BLAS kernels on GPUs),
StarPU not only takes care of offloading such kernels (and
implementing data coherency across the machine), but it also makes
sure the kernels are executed as efficiently as possible.

## What StarPU is not

StarPU is not a new language, and it does not extend existing languages either.
StarPU does not help to write computation kernels.

## (How) Could StarPU help me?

While StarPU will not make it easier to write computation kernels, it does
simplify their actual offloading as StarPU handle most low level aspects
transparently.

Obviously, it is crucial to have efficient kernels, but it must be noted that
the way those kernels are mapped and scheduled onto the computational resources
also affect the overall performance to a great extent.

StarPU is especially helpful when considering multiple heterogeneous processing
resources: statically mapping and synchronizing tasks in such a heterogeneous
environment is already very difficult, making it in a portable way is virtually
impossible. On the other hand, the scheduling capabilities of StarPU makes it
possible to easily exploit all processors at the same time while taking
advantage of their specificities in a portable fashion.

## Requirements

* `make`
* `gcc` (version >= 4.1)
* if `CUDA` support is enabled
  * `CUDA` (version >= 2.2)
  * `CUBLAS` (version >= 2.2)
* if `OpenCL` support is enabled
  * `AMD` SDK >= 2.3 if `AMD` driver is used
  * `CUDA` >= 3.2 if `NVIDIA` driver is used
* extra requirements for the `git` version (we usually use the Debian testing versions)
  * `autoconf` (version >= 2.60)
  * `automake`
  * `makeinfo`
  * `libtool` (version >= 2)

Remark: It is strongly recommended that you also install the hwloc library
   before installing StarPU. This permits StarPU to actually map the processing
   units according to the machine topology. For more details on hwloc, see
   http://www.open-mpi.org/projects/hwloc/ .

## Getting StarPU

StarPU is available on https://gitlab.inria.fr/starpu/starpu

The GIT repository access can be checked out with the following command.

```shell
$ git clone https://gitlab.inria.fr/starpu/starpu.git
```

## Building and Installing

### For git version only

Please skip this step if you are building from a tarball.

```shell
$ ./autogen.sh
```

### For all versions

```shell
$ mkdir build && cd build
$ ../configure
$ make
$ make install
```

### If building fails

Please post the commands you used to build, your whole build log, and the obtained `config.log` in the bug report.

### Windows build

StarPU can be built using MinGW or Cygwin.  To avoid the cygwin dependency,
we provide MinGW-built binaries.  The build process produces `libstarpu.dll`,
`libstarpu.def`, and `libstarpu.lib`, which should be enough to use it from e.g.
Microsoft Visual Studio.

Update the video drivers to the latest stable release available for your
hardware. Old ATI drivers (< 2.3) contain bugs that cause OpenCL support in
StarPU to hang or exhibit incorrect behaviour.

For details on the Windows build process, see the [INSTALL](https://gitlab.inria.fr/starpu/starpu/-/blob/master/INSTALL) file.

## Running StarPU Applications on Microsoft Visual C

Batch files are provided to run StarPU applications under Microsoft
Visual C. They are installed in `path_to_starpu/bin/msvc`.

To execute a StarPU application, you first need to set the environment
variable `STARPU_PATH`.

```shell
c:\....> cd c:\cygwin\home\ci\starpu\
c:\....> set STARPU_PATH=c:\cygwin\home\ci\starpu\
c:\....> cd bin\msvc
c:\....> starpu_open.bat starpu_simple.c
```

The batch script will run Microsoft Visual C with a basic project file
to run the given application.

The batch script `starpu_clean.bat` can be used to delete all
compilation generated files.

The batch script `starpu_exec.bat` can be used to compile and execute a
StarPU application from the command prompt.

```shell
c:\....> cd c:\cygwin\home\ci\starpu\
c:\....> set STARPU_PATH=c:\cygwin\home\ci\starpu\
c:\....> cd bin\msvc
c:\....> starpu_exec.bat ..\..\..\..\examples\basic_examples\hello_world.c

MSVC StarPU Execution
...
/out:hello_world.exe
...
Hello world (params = {1, 2.00000})
Callback function got argument 0000042
c:\....>
```

## Documentation

Doxygen documentation is available in `doc/doxygen`. If the doxygen
tools are available on the machine, pdf and html documentation can be
generated by running

```shell
$ make -C doc
```

The [documentation for the latest StarPU release](https://files.inria.fr/starpu/doc/html/) is available, as well as
the [documentation for the StarPU master branch](https://files.inria.fr/starpu/testing/master/doc/html/).

## Trying

Some examples ready to run are installed into `$prefix/lib/starpu/{examples,mpi}`

## Upgrade

To upgrade your source code from older version (there were quite a few
renamings), use the `tools/dev/rename.sh` script.

## Contribute

Contributions are welcome! Both on the
[main StarPU repository](https://gitlab.inria.fr/starpu/starpu)
and on the
[github StarPU mirror](https://github.com/starpu-runtime/starpu)

Please see [our contribution page](https://starpu.gitlabpages.inria.fr/involved.html) for details.

## Contact

For any questions regarding StarPU, please contact the starpu-devel
mailing-list at starpu-devel@inria.fr or browse
[the StarPU website](https://starpu.gitlabpages.inria.fr/).

You can also contact the devel team
[github](https://github.com/starpu-runtime/starpu/discussions) or
[discord](https://discord.gg/ERq6hvY)
