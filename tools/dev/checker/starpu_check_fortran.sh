# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2023-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

# display functions not available in fortran

for func in $(grep '^[a-z]* \<starpu_.*(.*)' include/*h | while read line
	      do
		  echo $line | sed 's/include.*.h://' | sed 's/(.*//' | awk '{print $NF}' | tr -d '*'
	      done)
do
    x=$(grep "$func(" include/*f90)
    if test -z "$x"
    then
	echo $func
    else
	true
	#echo $func
    fi
done
