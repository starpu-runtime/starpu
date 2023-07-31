#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

ok()
{
    return
    type=$1
    name=$2
    echo "$type ${greencolor}${name}${stcolor} is (maybe correctly) documented"
}

ko()
{
    type=$1
    name=$2
    echo "$type ${redcolor}${name}${stcolor} is not (or incorrectly) documented ($3)"
}

if [ "$1" == "--func" ] || [ "$1" == "" ]
then
    for f in $STARPU_H_FILES $SC_H_FILES
    do
	grep "(" $f | grep ';' | grep starpu | grep '^[a-z]' | grep -v typedef | grep -v '(\*' | while read line
	do
	    x=$(grep -F -B1 "$line" $f | head -1 | xargs)
	    fname=$(echo $line | awk -F'(' '{print $1}' | awk '{print $NF}' | tr -d '*')
	    if test "$x" == '*/'
	    then
		ok function $fname
	    else
		x=$(grep -l -F "$line" $f)
		ko function $fname "$x"
	    fi
	done
    done
fi

if [ "$1" == "--struct" ] || [ "$1" == "" ] ; then
    starpu=$(grep "^struct starpu_[a-z_]*$" $STARPU_H_FILES | awk '{print $NF}')
    sc=$(grep "^struct sc_[a-z_]*$" $SC_H_FILES | awk '{print $NF}')
    for o in $starpu $sc ; do
	hfile=$(grep -l "^struct ${o}$" $STARPU_H_FILES $SC_H_FILES)
	x=$(grep -B1 "^struct ${o}$" $hfile | head -1 | xargs)
	if test "$x" == '*/'
	then
	    ok "struct" ${o}
	else
	    x=$(grep -l "^struct ${o}$" $hfile)
	    ko "struct" ${o} "$x"
	fi
    done
    echo
fi

if [ "$1" == "--enum" ] || [ "$1" == "" ] ; then
    starpu=$(grep "^enum starpu_[a-z_]*$" $STARPU_H_FILES | awk '{print $NF}')
    sc=$(grep "^enum sc_[a-z_]*$" $SC_H_FILES | awk '{print $NF}')
    for o in $starpu $sc ; do
	hfile=$(grep -l "^enum ${o}$" $STARPU_H_FILES $SC_H_FILES)
	x=$(grep -B1 "^enum ${o}$" $hfile | head -1 | xargs)
	if test "$x" == '*/'
	then
	    ok "enum" ${o}
	else
	    x=$(grep -l "^enum ${o}$" $hfile)
	    ko "enum" ${o} "$x"
	fi
    done
    echo
fi

if [ "$1" == "--macro" ] || [ "$1" == "" ] ; then
    macros=$(grep "define\b" $STARPU_H_FILES $SC_H_FILES |grep -v deprecated|grep "#" | grep -v "__" | sed 's/#[ ]*/#/g' | awk '{print $2}' | awk -F'(' '{print $1}' | grep -i starpu | sort|uniq)
    for o in $macros ; do
	hfile=$(grep -l "define\b ${o}" $STARPU_H_FILES $SC_H_FILES)
	x=$(grep -B1 "define\b ${o}" $hfile | head -1 | xargs)
	if test "$x" == '*/'
	then
	    ok "define" ${o}
	else
	    x=$(grep -l "define\b ${o}" $hfile)
	    ko "define" ${o} "$x"
	fi
    done
    echo
fi

if [ "$1" == "--var" ] || [ "$1" == "" ] ; then
    variables=$(grep -rs -E "(getenv|get_env)" $SRC| tr ' ' '\012'|grep -E "(getenv|get_env)" | grep "\"" | sed 's/.*("//' | sed 's/").*//'|tr -d '",'|sort|uniq)
    for variable in $variables ; do
	x=$(grep "$variable" $dirname/../chapters/starpu_installation/environment_variables.doxy | grep "\\anchor")
	if test "$x" == "" ; then
	    ko "variable" $variable
	else
	    ok "variable" $variable
	fi
    done
fi
