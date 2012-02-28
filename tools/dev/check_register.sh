#!/bin/bash

# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011  Centre National de la Recherche Scientifique
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
datacolor=$(tput setaf 2)
filecolor=$(tput setaf 1)

process_file()
{
    datas=$(grep "data_register(" $f| awk -F',' '{print $1}' | awk -F'(' '{print $2}' | tr -d '&' | sed 's/\[/\\\[/g' | sed 's/\]/\\\]/g' | sed 's/\*/\\\*/g')
    for data in $datas ; do
	x=$(grep "data_unregister($data" $1)
	if test "$x" == "" ; then
	    x=$(grep "data_unregister_no_coherency($data" $1)
	    if test "$x" == "" ; then
		echo "Error. File <${filecolor}$1${stcolor}>. Handle <${datacolor}$data${stcolor}> is not unregistered"
	    fi
	fi
    done
}

for f in $(find tests -type f -not -path "*svn*") ; do process_file $f ; done
for f in $(find examples -type f -not -path "*svn*") ; do process_file $f ; done
