#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#grep --exclude-dir=.git --binary-files=without-match -rsF "\ref" $dirname/../chapters|grep -v "\\ref [a-zA-Z]"
#echo continue && read

GREP="grep --exclude-dir=.git --binary-files=without-match -rsF"

REFS=$($GREP "\ref" $dirname/../chapters| tr ':' '\012' | tr '.' '\012'  | tr ',' '\012'  | tr '(' '\012' | tr ')' '\012' | tr ' ' '\012'|grep -F '\ref' -A1 | grep -v '^--$' | sed 's/\\ref/=\\ref/' | tr '\012' ':' | tr '=' '\012' | sort | uniq)
find $dirname/../chapters -name "*doxy" -exec cat {} \; > /tmp/DOXYGEN_$$
cat $dirname/../refman.tex >> /tmp/DOXYGEN_$$

for r in $REFS
do
    ref=$(echo $r | sed 's/\\ref:\(.*\):/\1/')
    n=$($GREP -crs "section $ref" /tmp/DOXYGEN_$$)
    if test $n -eq 0
    then
	n=$($GREP -crs "anchor $ref" /tmp/DOXYGEN_$$)
	if test $n -eq 0
	then
	    n=$($GREP -crs "ingroup $ref" /tmp/DOXYGEN_$$)
	    if test $n -eq 0
	    then
		n=$($GREP -crs "def $ref" /tmp/DOXYGEN_$$)
		if test $n -eq 0
		then
		    n=$($GREP -crs "struct $ref" /tmp/DOXYGEN_$$)
		    if test $n -eq 0
		    then
			if test $n -eq 0
			then
			    n=$($GREP -crs "label{$ref" /tmp/DOXYGEN_$$)
			    if test $n -eq 0
			    then
				echo $ref missing
			    fi
			fi
		    fi
		fi
	    fi
	fi
    fi
done
