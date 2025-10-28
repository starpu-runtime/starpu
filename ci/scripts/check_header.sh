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

code=0

TOOLSDIR=$(dirname $0)/../../tools
$TOOLSDIR/dev/checker/starpu_check_copyright.sh
rc=$?
if [ $rc -eq 0 ]
then
    echo "Check header: SUCCESS"
else
    echo "Check header: FAILED"
    code=1
fi

exit $code
