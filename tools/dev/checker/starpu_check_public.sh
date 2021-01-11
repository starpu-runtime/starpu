#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

test_()
{
    INCLUDE=$1
    INCLUDE_FILES=$(find $INCLUDE -name '*.h')
    shift
    echo "Check include files in directory <$INCLUDE> against $*"
    ok=1

    functions=$(spatch -very_quiet -sp_file doc/doxygen/dev/starpu_funcs.cocci $INCLUDE_FILES)
    for func in $functions ; do
	fname=$(echo $func|awk -F ',' '{print $1}')
	location=$(echo $func|awk -F ',' '{print $2}')
	x=$(grep -rs "$fname" $*)
	if test "$x" == "" ; then
	    ok=0
	    echo "function ${redcolor}${fname}${stcolor} at location ${redcolor}$location${stcolor} is not used in any examples or tests"
	fi
    done

    echo

    structs=$(grep "struct starpu" $INCLUDE_FILES | grep -v "[;|,|(|)]" | awk '{print $2}')
    for struct in $structs ; do
	x=$(grep -rs "struct $struct" $*)
	if test "$x" == "" ; then
	    ok=0
	    echo "struct ${redcolor}${struct}${stcolor} is not used in any examples or tests"
	fi
    done

    echo

    enums=$(grep "enum starpu" $INCLUDE_FILES | grep -v "[;|,|(|)]" | awk '{print $2}')
    for enum in $enums ; do
	x=$(grep -rs "enum $enum" $*)
	if test "$x" == "" ; then
	    ok=0
	    echo "enum ${redcolor}${enum}${stcolor} is not used in any examples or tests"
	fi
    done

    echo

    macros=$(grep "define\b" $INCLUDE_FILES|grep -v deprecated|grep "#" | grep -v "__" | sed 's/#[ ]*/#/g' | awk '{print $2}' | awk -F'(' '{print $1}' | sort|uniq)
    for macro in $macros ; do
	x=$(grep -rs "$macro" $*)
	if test "$x" == "" ; then
	    ok=0
	    echo "macro ${redcolor}${macro}${stcolor} is not used in any examples or tests"
	fi
    done

    if test "$ok" == "1" ; then
	echo "All OK"
    fi
}

test_ include examples tests mpi/src starpufft tools src/sched_policies
test_ mpi/include mpi/src mpi/examples
