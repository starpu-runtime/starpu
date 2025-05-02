#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2013-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

EXCLUDE=$(dirname $0)/starpu_check_copyright_exclude.txt
VERBOSE=true
if test "$1" = "-v"
then
    VERBOSE=echo
    shift
fi

check_copyright()
{
    $VERBOSE "    check copyright"
    copyright=$(grep "StarPU is free software" $1 2>/dev/null)
    if test -z "$copyright"
    then
	echo "File $1 does not include a proper copyright"
	git log $f | grep '^Author:' | sort | uniq
        nberr=$(( nberr + 1 ))
    fi
}

check_header_define()
{
    filename=$1
    basename=$(basename $filename)

    if ! echo $filename | grep -q include
    then
	return
    fi

    case $basename in
	omp.h)
	    ;;
	pthread.h)
	    ;;
	semaphore.h)
	    ;;
        *.h)
	    $VERBOSE "    check define"
            n=$(basename $basename .h | awk '{print toupper($0)}')
            macro="__${n}_H__"
            err=0

            toto=$(grep "#ifndef .*$macro" $filename)
            ret=$?
            err=$((err + ret))

            if [ $ret -eq 0 ]
            then
                macro=$(grep "#ifndef" $filename | sed 's/#ifndef //')
            fi
            toto=$(grep "#define $macro" $filename)
            ret=$?
            err=$((err + ret))

            toto=$(grep "#endif /\* $macro \*/" $filename)
            ret=$?
            err=$((err + ret))

            if [ $err -ne 0 ]
            then
		echo "File $1 does not properly define $macro"
                #grep "#ifndef" $filename
                #grep "#define" $filename
                #grep "#endif"  $filename
                nberr=$(( nberr + 1 ))
            fi
            ;;
        *)
    esac
}

PARAMS=${*:-.}
files=$(for x in $PARAMS
	do
	    git ls-files $x |
		grep -v uthash.h |
		grep -v ax_cxx_compile_stdcxx.m4 |
		grep -v pkg.m4 |
		grep -v rbtree |
		grep -v gcc-plugin |
		grep -v min-dgels |
		grep -v starpu-top |
		grep -v SobolQRNG |
		grep -v socl/src/CL |
		grep -v ocl_icd.h |
		grep -v socl.icd.in |
		grep -v starpujni/cmake |
		grep -v starpujni/src |
		grep -v starpujni/scripts |
		grep -v tools/gpus |
		grep -v cproject.in |
		grep -v build-aux |
		grep -v tools/perfmodels/sampling |
		grep -v .png |
		grep -v .gitignore |
		grep -vi issue_template |
		grep -v .out |
		grep -v .xml |
		grep -v .maxj |
		grep -v .dat
	done)

nberr=0
for f in $files
do
    if grep -q $f ${EXCLUDE}
    then
	continue
    fi

    $VERBOSE "------ $f --------"
    check_copyright $f
    check_header_define $f
done

if [ $nberr -gt 0 ]
then
    echo "${nberr} mistakes have been found in the StarPU files."
    exit 1
else
    echo "No mistake found in the StarPU files."
    exit 0
fi
