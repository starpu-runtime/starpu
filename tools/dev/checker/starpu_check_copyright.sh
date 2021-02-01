#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
REP=${1:-.}

find $REP -not -path "*build*" -not -path "*.git*" -not -path "*tools/perfmodels/sampling*" -not -path "*starpu-top*"  -not -path "*min-dgels*" -not -name ".gitignore" -not -name "*.eps"  -not -name "*.pdf" -not -name "*.png" -not -path "*.deps*" -type f > /tmp/list_$$

for f in $(cat /tmp/list_$$)
do
    copyright=$(grep "StarPU is free software" $f 2>/dev/null)
    if test -z "$copyright"
    then
	echo "File $f does not include a proper copyright"
	git log $f | grep '^Author:' | sort | uniq
    fi
done
