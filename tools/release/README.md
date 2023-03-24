<!--
StarPU --- Runtime system for heterogeneous multicore architectures.
Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
StarPU is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at
your option) any later version.
StarPU is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License in COPYING.LGPL for more details.
-->

The makefile in this directory should be used to test the compilation
and execution of StarPU examples against an installed version of
StarPU.

For example, if StarPU is installed in

```
STARPU_INST=$HOME/softs/starpu-1.4
```

and the source code of StarPU is in

```
STARPU_SRC=$HOME/src/starpu/master
```

one first need to call the following script

```
source $STARPU_INST/bin/starpu_env
```

and then call

```
make STARPU=starpu-1.4 EXAMPLE=$STARPU_SRC/examples
```

to produce the executables.

Examples using an old StarPU API can also be tested, for example the branch 1.0

```
make STARPU=starpu-1.0 EXAMPLE=$HOME/src/starpu/branches/starpu-1.0/examples/
```

Note the variable STARPU is set to starpu-1.0 to use the 1.0 API.
