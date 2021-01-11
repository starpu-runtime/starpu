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
dirname=$(dirname $0)

x=$(grep ingroup $dirname/../chapters/api/*.doxy $dirname/../chapters/api/sc_hypervisor/*.doxy |awk -F':' '{print $2}'| awk 'NF != 2')
if test -n "$x" ; then
    echo Errors on group definitions
    echo $x
fi

echo
echo "Defined groups"
grep ingroup $dirname/../chapters/api/*.doxy $dirname/../chapters/api/sc_hypervisor/*.doxy|awk -F':' '{print $2}'| awk 'NF == 2'|sort|uniq
echo

for f in $dirname/../../../build/doc/doxygen/latex/*tex ; do
    x=$(grep $(basename $f .tex) $dirname/../refman.tex)
    if test -z "$x" ; then
	echo Error. $f not included in refman.tex
    fi
done

