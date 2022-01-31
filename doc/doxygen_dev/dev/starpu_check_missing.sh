#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

stcolor=$(tput sgr0)
redcolor=$(tput setaf 1)
greencolor=$(tput setaf 2)

dirname=$(realpath $(dirname $0))

ok()
{
    type=$1
    name=$2
    echo "$type ${greencolor}${name}${stcolor} is in doxygen-config.cfg.in"
}

ko()
{
    type=$1
    name=$2
    #echo "$type ${redcolor}${name}${stcolor} is missing from doxygen-config.cfg.in"
    echo $name
}

for d in src mpi/src starpurm/src
do
    cd $dirname/../../../$d
    for f in $(find -name "*.h")
    do
	ff=$(echo $f | cut -b3-)
	x=$(grep -c $ff $dirname/../doxygen-config.cfg.in)
	if test "$x" == "0"
	then
	    ko file $d/$ff
	#else
	#    ok file $d/$ff
	fi
    done
done

cd $dirname/../../../build/doc/doxygen_dev/latex
for f in $(find -name "*8h.tex")
do
    ff=$(basename $(echo $f | cut -b3-) ".tex")
    x=$(grep -c $ff refman.tex)
    if test "$x" == "0"
    then
	ko file $ff
    fi
done

