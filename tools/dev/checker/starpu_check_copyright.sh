#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
EXCLUDE=$(dirname $0)/starpu_check_copyright_exclude.txt

    git ls-files $REP |
    grep -v uthash.h |
    grep -v ax_cxx_compile_stdcxx.m4 |
    grep -v pkg.m4 |
    grep -v rbtree |
    grep -v gcc-plugin |
    grep -v min-dgels |
    grep -v starpu-top |
    grep -v SobolQRNG |
    grep -v socl/src/CL |
    grep -v ocl_icd.h |
    grep -v socl.icd.in |
    grep -v starpujni/cmake |
    grep -v starpujni/src |
    grep -v starpujni/scripts |
    grep -v tools/gpus |
    grep -v cproject.in |
    grep -v build-aux |
    grep -v tools/perfmodels/sampling |
    grep -v .png |
    grep -v .gitignore |
    grep -vi issue_template |
    grep -v .out |
    grep -v .xml |
    grep -v .maxj |
    grep -v .dat > /tmp/list_$$

for f in $(cat /tmp/list_$$)
do
    if grep -q $f ${EXCLUDE}
    then
	continue
    fi
    copyright=$(grep "StarPU is free software" $f 2>/dev/null)
    if test -z "$copyright"
    then
	echo "File $f does not include a proper copyright"
	git log $f | grep '^Author:' | sort | uniq
    fi
done
