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
EXEC=$(basename $0 .sh)
DIRNAME=$(dirname $0)

CLIMIT=$(ulimit -c)
if [ "$CLIMIT" = unlimited ]
then
	# valgrind cores are often *huge*, 100MB will already be quite big...
	ulimit -c 100000
fi

if test "$EXEC" == "valgrind"
then
    RUN="valgrind --track-origins=yes --show-reachable=yes --leak-check=full --errors-for-leak-kinds=all --show-leak-kinds=all --error-exitcode=42"
elif test "$EXEC" == "valgrind_xml"
then
    mkdir -p ${DIRNAME}/../../../valgrind
    XML_FILE=$(mktemp -p ${DIRNAME}/../../../valgrind starpu-valgrind_XXXXXXXXXX.xml)
    RUN="valgrind --track-origins=yes --show-reachable=yes --leak-check=full --errors-for-leak-kinds=all --show-leak-kinds=all --xml=yes --xml-file=${XML_FILE}"
else
    RUN="valgrind --tool=$EXEC --error-exitcode=42"
fi
SUPPRESSIONS=$(for f in $(dirname $0)/*.suppr /usr/share/hwloc/hwloc-valgrind.supp; do if test -f $f ; then echo "--suppressions=$f" ; fi ; done)

$RUN --keep-debuginfo=yes --num-callers=42 --error-limit=no --gen-suppressions=all $SUPPRESSIONS $*
