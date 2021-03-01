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

DIRS="$dirname/../../../include $dirname/../../../mpi/include $dirname/../../../starpurm/include $dirname/../../../sc_hypervisor/include"
echo "Defined groups"
groups=""
for d in $DIRS
do
    echo Checking $d
    gg=$(grep -rs defgroup $d | awk '{print $3}')
    echo $gg
    groups=$(echo $groups $gg)
done
for g in $groups
do
    gg=$(echo $g | sed 's/_/__/g')
    x=$(grep $gg $dirname/../refman.tex)
    if test -z "$x"
    then
	echo "Error. Group $g not included in refman.tex"
    fi
done
echo

for d in $DIRS
do
    for f in $(find $d -name "*.h")
    do
	ff=$(echo $f  | awk -F'/' '{print $NF}')
	x=$(grep $ff $dirname/../doxygen-config.cfg.in)
	if test -z "$x"
	then
	    echo Error. $f not included in doxygen-config.cfg.in
	fi
	x=$(grep $ff $dirname/../chapters/520_files.doxy)
	if test -z "$x"
	then
	    echo Error. $f not included in 520_files.doxy
	fi
    done
done
echo

for p in starpu sc__hypervisor
do
    for f in $dirname/../../../build/doc/doxygen/latex/${p}*tex
    do
	x=$(grep $(basename $f .tex) $dirname/../refman.tex)
	if test -z "$x"
	then
	    echo Error. $f not included in refman.tex
	fi
    done
done

