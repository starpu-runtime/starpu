#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
# Note: expects Coccinelle's spatch command n the PATH
# See: http://coccinelle.lip6.fr/

stcolor=$(tput sgr0)
redcolor=$(tput setaf 1)
greencolor=$(tput setaf 2)

dirname=$(dirname $0)

STARPU_H_FILES=$(find $dirname/../../../include $dirname/../../../mpi/include -name '*.h')
SC_H_FILES=$(find $dirname/../../../sc_hypervisor/include -name '*.h')
SRC="$dirname/../../../src $dirname/../../../mpi/src $dirname/../../../sc_hypervisor/src"

if [ "$1" == "--starpu" ]
then
    SC_H_FILES="$0"
    shift
else
    if [ "$1" == "--sc" ]
    then
	STARPU_H_FILES="$0"
	shift
    fi
fi

if [ "$1" == "--func" ] || [ "$1" == "" ] ; then
    starpu_functions=$(spatch -very_quiet -sp_file $dirname/starpu_funcs.cocci $STARPU_H_FILES)
    sc_functions=$(spatch -very_quiet -sp_file $dirname/sc_funcs.cocci $SC_H_FILES)
    for func in $starpu_functions $sc_functions ; do
	fname=$(echo $func|awk -F ',' '{print $1}')
	location=$(echo $func|awk -F ',' '{print $2}')
	x=$(grep "$fname(" $dirname/../chapters/api/*.doxy | grep "\\fn")
	if test "$x" == "" ; then
	    echo "function ${redcolor}${fname}${stcolor} at location ${redcolor}$location${stcolor} is not (or incorrectly) documented"
	    #	else
	    #		echo "function ${greencolor}${fname}${stcolor} at location $location is correctly documented"
	fi
    done
    echo
fi

if [ "$1" == "--struct" ] || [ "$1" == "" ] ; then
    starpu_structs=$(grep "struct starpu" $STARPU_H_FILES | grep -v "[;|,|(|)]" | awk '{print $2}')
    sc_structs=$(grep "struct sc" $SC_H_FILES | grep -v "[;|,|(|)]" | awk '{print $2}')
    for struct in $starpu_structs $sc_structs ; do
	x=$(grep -F "\\struct $struct" $dirname/../chapters/api/*.doxy)
	if test "$x" == "" ; then
	    echo "struct ${redcolor}${struct}${stcolor} is not (or incorrectly) documented"
	fi
    done
    echo
fi

if [ "$1" == "--enum" ] || [ "$1" == "" ] ; then
    starpu_enums=$(grep "enum starpu" $STARPU_H_FILES | grep -v "[;|,|(|)]" | awk '{print $2}')
    sc_enums=$(grep "enum starpu" $SC_H_FILES | grep -v "[;|,|(|)]" | awk '{print $2}')
    for enum in $starpu_enums $sc_enums ; do
	x=$(grep -F "\\enum $enum" $dirname/../chapters/api/*.doxy)
	if test "$x" == "" ; then
	    echo "enum ${redcolor}${enum}${stcolor} is not (or incorrectly) documented"
	fi
    done
    echo
fi

if [ "$1" == "--macro" ] || [ "$1" == "" ] ; then
    macros=$(grep "define\b" $STARPU_H_FILES $SC_H_FILES |grep -v deprecated|grep "#" | grep -v "__" | sed 's/#[ ]*/#/g' | awk '{print $2}' | awk -F'(' '{print $1}' | sort|uniq)
    for macro in $macros ; do
	x=$(grep -F "\\def $macro" $dirname/../chapters/api/*.doxy)
	if test "$x" == "" ; then
	    echo "macro ${redcolor}${macro}${stcolor} is not (or incorrectly) documented"
	fi
    done
    echo
fi

if [ "$1" == "--var" ] || [ "$1" == "" ] ; then
    variables=$(grep -rs -E "(getenv|get_env)" $SRC| tr ' ' '\012'|grep -E "(getenv|get_env)" | grep "\"" | sed 's/.*("//' | sed 's/").*//'|tr -d '",'|sort|uniq)
    for variable in $variables ; do
	x=$(grep "$variable" $dirname/../chapters/40environment_variables.doxy | grep "\\anchor")
	if test "$x" == "" ; then
	    echo "variable ${redcolor}${variable}${stcolor} is not (or incorrectly) documented"
	fi
    done
fi

