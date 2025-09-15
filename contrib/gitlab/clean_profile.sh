#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2025-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
if test "x$1" = "x"
then
    echo Argument missing
    exit 1
fi

set +e
echo "Cleaning $1"
for x in $(seq 1 5)
do
    kill -9 $(ps fx | grep $1 | grep -v gitlab-runner | grep -v sshd | grep -v clean_profile | grep -v grep | awk '{print $1}')
done
