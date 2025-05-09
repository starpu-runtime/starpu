<!---
 StarPU --- Runtime system for heterogeneous multicore architectures.

 Copyright (C) 2020-2025    University of Bordeaux, CNRS (LaBRI UMR 5800), Inria

 StarPU is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or (at
 your option) any later version.

 StarPU is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

 See the GNU Lesser General Public License in COPYING.LGPL for more details.
-->

# Installing Julia

Julia version 1.3+ is required and can be downloaded from
https://julialang.org/downloads/.

# Installing StarPU module for Julia

First, build the `jlstarpu_c_wrapper` library:

```shell
$ make
```

Then, you need to add the `lib/` directory to your library path and the `julia/`
directory to your Julia load path:

```shell
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/src/.lib
$ export JULIA_LOAD_PATH=$PWD/src:$JULIA_LOAD_PATH
```

This step can also be done by sourcing the setenv.sh script:

```shell
$ . setenv.sh
```

# Running Examples

You can find several examples in the `examples/` directory.

For each example `X`, three versions are provided:

* `X.c`: Original C+starpu code
* `X_native.jl`: Native Julia version (without StarPU)
* `X.jl`: Julia version using StarPU

To run the original C+StarPU code:
```shell
$ make cstarpu.dat
```

To run the native Julia version:
```shell
$ make julia_native.dat
```

To run the Julia version using StarPU:
```shell
$ make julia_generatedc.dat
```




