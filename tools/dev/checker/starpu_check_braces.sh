#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2017                                CNRS
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
SHOW=cat
SHOW=less

DIRS="tools src tests examples mpi"
for d in ${1:-$DIRS}
do
    for ext in c h cl cu doxy
    do
	grep -rsn "{" $d |grep ".${ext}:" | grep -v "}" | grep -v ".${ext}:[0-9]*:[[:space:]]*{$" > /tmp/braces
	if test -s /tmp/braces
	then
	    $SHOW /tmp/braces
	fi
	grep -rsn "}" $d |grep ".${ext}:" | grep -v "{" | grep -v "};" | grep -v ".${ext}:[0-9]*:[[:space:]]*};*$" > /tmp/braces
	if test -s /tmp/braces
	then
	    $SHOW /tmp/braces
	fi
    done
done
