#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013,2014,2016,2017,2019                      CNRS
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

echo "Defined groups"
groups=$(grep -rs defgroup $dirname/../../../include | awk '{print $3}')
echo $groups
echo
for g in $groups
do
    gg=$(echo $g | sed 's/_/__/g')
    x=$(grep $gg $dirname/../refman.tex)
    if test -z "$x"
    then
	echo "Error. Group $g not included in refman.tex"
    fi
done

for f in $(find $dirname/../../../include -name "starpu*.h")
do
    ff=$(echo $f  | awk -F'/' '{print $NF}')
    x=$(grep $ff $dirname/../doxygen-config.cfg.in)
    if test -z "$x"
    then
	echo Error. $f not included in doxygen-config.cfg.in
    fi
done

for f in $dirname/../../../build/doc/doxygen/latex/starpu*tex
do
    x=$(grep $(basename $f .tex) $dirname/../refman.tex)
    if test -z "$x"
    then
	echo Error. $f not included in refman.tex
    fi
done

