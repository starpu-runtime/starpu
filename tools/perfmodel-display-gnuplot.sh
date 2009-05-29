#!/bin/bash

#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

function gnuplot_symbol()
{
symbol=$1

echo "Display symbol $symbol"

# TODO check return value $? of perfmodel-display to ensure we have valid data
cuda_a=`./perfmodel-display -s $symbol -a cuda -p a`
cuda_b=`./perfmodel-display -s $symbol -a cuda -p b`
cuda_c=`./perfmodel-display -s $symbol -a cuda -p c`

cuda_alpha=`./perfmodel-display -s $symbol -a cuda -p alpha`
cuda_beta=`./perfmodel-display -s $symbol -a cuda -p beta`

cuda_debug=`./perfmodel-display -s $symbol -p path-file-debug -a cuda`

echo "CUDA : y = $cuda_a * size ^ $cuda_b + $cuda_c"
echo "CUDA : y = $cuda_alpha * size ^ $cuda_beta"
echo "CUDA : debug file $cuda_debug"

core_a=`./perfmodel-display -s $symbol -a core -p a`
core_b=`./perfmodel-display -s $symbol -a core -p b`
core_c=`./perfmodel-display -s $symbol -a core -p c`

core_alpha=`./perfmodel-display -s $symbol -a core -p alpha`
core_beta=`./perfmodel-display -s $symbol -a core -p beta`

core_debug=`./perfmodel-display -s $symbol -p path-file-debug -a core`

echo "CORE : y = $core_a * size ^ $core_b + $core_c"
echo "CORE : y = $core_alpha * size ^ $core_beta"
echo "CORE : debug file $core_debug"

gnuplot > /dev/null << EOF
set term postscript eps enhanced color
set output "model_$symbol.eps"

set logscale x
set logscale y

plot $core_a * ( x ** $core_b ) + $core_c title "core (non linear)" ,\
	$cuda_a * ( x ** $cuda_b ) + $cuda_c title "cuda (non linear)" ,\
	"$core_debug" usi 2:3 title "core measured" ,\
	"$cuda_debug" usi 2:3 title "cuda measured"

EOF

}

for symbol in $@
do
	gnuplot_symbol $symbol
done
