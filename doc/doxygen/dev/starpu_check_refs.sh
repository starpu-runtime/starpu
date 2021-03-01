#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#grep --exclude-dir=.git --binary-files=without-match -rsF "\ref" $dirname/../chapters|grep -v "\\ref [a-zA-Z]"
#echo continue && read

GREP="grep --exclude-dir=.git --binary-files=without-match -rsF"

REFS=$($GREP "\ref" $dirname/../chapters| tr ':' '\012' | tr '.' '\012'  | tr ',' '\012'  | tr '(' '\012' | tr ')' '\012' | tr ' ' '\012'|grep -F '\ref' -A1 | grep -v '^--$' | sed 's/\\ref/=\\ref/' | tr '\012' ':' | tr '=' '\012' | sort | uniq)
find $dirname/../chapters -name "*doxy" -exec cat {} \; > /tmp/DOXYGEN_$$
cat $dirname/../refman.tex >> /tmp/DOXYGEN_$$
find $dirname/../../../include -name "*h" -exec cat {} \; >> /tmp/DOXYGEN_$$
find $dirname/../../../starpurm/include -name "*h" -exec cat {} \; >> /tmp/DOXYGEN_$$
find $dirname/../../../mpi/include -name "*h" -exec cat {} \; >> /tmp/DOXYGEN_$$
find $dirname/../../../sc_hypervisor/include -name "*h" -exec cat {} \; >> /tmp/DOXYGEN_$$

stcolor=$(tput sgr0)
redcolor=$(tput setaf 1)
greencolor=$(tput setaf 2)

for r in $REFS
do
    ref=$(echo $r | sed 's/\\ref:\(.*\):/\1/')
    if test -n "$ref"
    then
	#echo "ref $ref"
	for keyword in "section " "anchor " "ingroup " "defgroup " "def " "struct " "label{"
	do
	    n=$($GREP -crs "${keyword}${ref}" /tmp/DOXYGEN_$$)
	    if test $n -ne 0
	    then
		break
	    fi
	done
	if test $n -eq 0
	then
	    echo "${redcolor}$ref${stcolor} is missing"
	else
	    true
	    #echo "${greencolor}$ref${stcolor} is ok"
	fi
    fi
done
