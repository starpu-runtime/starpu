#!/bin/bash
# Note: expects Coccinelle's spatch command n the PATH
# See: http://coccinelle.lip6.fr/

# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011, 2012, 2013 Centre National de la Recherche Scientifique
# Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique
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

stcolor=$(tput sgr0)
redcolor=$(tput setaf 1)
greencolor=$(tput setaf 2)

H_FILES=$(find ../../include ../../mpi/include -name '*.h')

functions=$(spatch -very_quiet -sp_file ./dev/starpu_funcs.cocci $H_FILES)
for func in $functions ; do
	fname=$(echo $func|awk -F ',' '{print $1}')
	location=$(echo $func|awk -F ',' '{print $2}')
	x=$(grep "$fname(" chapters/api/*.doxy | grep "\\fn")
	if test "$x" == "" ; then
		echo "function ${redcolor}${fname}${stcolor} at location ${redcolor}$location${stcolor} is not (or incorrectly) documented"
#	else
#		echo "function ${greencolor}${fname}${stcolor} at location $location is correctly documented"
	fi
done

echo

structs=$(grep "struct starpu" $H_FILES | grep -v "[;|,|(|)]" | awk '{print $2}')
for struct in $structs ; do
    x=$(grep -F "\\struct $struct" chapters/api/*.doxy)
    if test "$x" == "" ; then
	echo "struct ${redcolor}${struct}${stcolor} is not (or incorrectly) documented"
    fi
done

echo

enums=$(grep "enum starpu" $H_FILES | grep -v "[;|,|(|)]" | awk '{print $2}')
for enum in $enums ; do
    x=$(grep -F "\\enum $enum" chapters/api/*.doxy)
    if test "$x" == "" ; then
	echo "enum ${redcolor}${enum}${stcolor} is not (or incorrectly) documented"
    fi
done

echo

macros=$(grep "define\b" $H_FILES |grep -v deprecated|grep "#" | grep -v "__" | sed 's/#[ ]*/#/g' | awk '{print $2}' | awk -F'(' '{print $1}' | sort|uniq)
for macro in $macros ; do
    x=$(grep -F "\\def $macro" chapters/api/*.doxy)
    if test "$x" == "" ; then
	echo "macro ${redcolor}${macro}${stcolor} is not (or incorrectly) documented"
    fi
done

echo

variables=$(grep --exclude-dir=.svn -rs -E "(getenv|get_env)" src/| tr ' ' '\012'|grep -E "(getenv|get_env)" | grep "\"" | sed 's/.*("//' | sed 's/").*//'|sort|uniq)
for variable in $variables ; do
    x=$(grep "$variable" chapters/environment_variables.doxy | grep "\\anchor")
    if test "$x" == "" ; then
	echo "variable ${redcolor}${variable}${stcolor} is not (or incorrectly) documented"
    fi
done

