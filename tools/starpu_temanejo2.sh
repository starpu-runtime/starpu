#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
d=${AYUDAME2_INSTALL_DIR?}
cmd=${1?"usage: $0 <cmd> [args*]"}
shift
if test ! -r ayudame.cfg; then
	echo "warning: no 'ayudame.cfg' file found in current working directory, an example is available in <STARPU_INSTALL_DIR>/share/starpu/ayudame.cfg"
fi
PATH=$d/bin:$PATH
LD_LIBRARY_PATH=$d/lib:$LD_LIBRARY_PATH
PYTHONPATH=$d/lib/python2.7/site-packages:$PYTHONPATH
export PATH LD_LIBRARY_PATH PYTHONPATH
$d/bin/Temanejo2 -p 8888 -d 8889 -P $d/lib/libayudame.so -L $d/lib -A $cmd "$@"
