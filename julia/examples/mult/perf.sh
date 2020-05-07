#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

stride=72
#stride=4

export STARPU_NOPENCL=0
export STARPU_SCHED=dmda
export STARPU_CALIBRATE=1

rm -f ./cstarpu.dat julia_generatedc.dat julia_native.dat julia_calllib.dat

$(dirname $0)/mult $stride > ./cstarpu.dat
$(dirname $0)/../execute.sh mult/mult.jl $stride julia_generatedc.dat
$(dirname $0)/../execute.sh mult/mult_native.jl $stride julia_native.dat
$(dirname $0)/../execute.sh -calllib mult/cpu_mult.c mult/mult.jl $stride julia_calllib.dat

(
    cat <<EOF
set output "comparison.pdf"
set term pdf
plot "julia_native.dat" w l,"cstarpu.dat" w l,"julia_generatedc.dat" w l,"julia_calllib.dat" w l
EOF
) | gnuplot
