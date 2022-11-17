#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

set -e

./contrib/ci.inria.fr/job-0-tarball.sh

tarball=$(ls -tr starpu-*.tar.gz | tail -1)

if test -z "$tarball"
then
    echo Error. No tar.gz file
    ls
    pwd
    exit 1
fi

if test ! -f starpu.pdf
then
    echo Error. No documentation file
    ls
    pwd
    exit 1
fi
