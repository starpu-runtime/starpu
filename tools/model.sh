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


trace_model()
{
	inputfile=$1
	
	coreentries=`head -1 $inputfile`
	gpuentries=`head -2 $inputfile|tail -1`
	
	coremodel=`head -3 $inputfile|tail -1`
	gpumodel=`head -4 $inputfile|tail -1`
	
	a_core=`cut -f 1 $inputfile| head -5|tail -1`
	b_core=`cut -f 2 $inputfile| head -5|tail -1`
	c_core=`cut -f 3 $inputfile| head -5|tail -1`
	
	a_gpu=`cut -f 1 $inputfile| head -6|tail -1`
	b_gpu=`cut -f 2 $inputfile| head -6|tail -1`
	c_gpu=`cut -f 3 $inputfile| head -6|tail -1`

	alpha_core=`cut -f 5 $inputfile| head -3|tail -1` 
	alpha_gpu=`cut -f 5 $inputfile| head -4|tail -1` 
	
	beta_core=`cut -f 6 $inputfile| head -3|tail -1` 
	beta_gpu=`cut -f 6 $inputfile| head -4|tail -1` 
	
	tail -$(($gpuentries + $coreentries)) $inputfile | head -$(($coreentries)) |cut -f 2-4 > $inputfile.core
	tail -$(($gpuentries)) $inputfile | cut -f 2-4> $inputfile.gpu
	
	echo "pouet $coreentries gpu $gpuentries toot"
	
	echo "coremodel $alpha_core * size ^ $beta_core"
	echo "gpumodel $alpha_gpu * size ^ $beta_gpu"
	
	gpfile=$inputfile.gp
	
	echo "#!/usr/bin/gnuplot -persist" 		> $gpfile
	echo "set term postscript eps enhanced color" 	>> $gpfile
	echo "set logscale x"				>> $gpfile 
	echo "set logscale y"				>> $gpfile 
	echo "set key left top"				>> $gpfile 
	echo "set title \"$inputfile\""			>> $gpfile 
	echo "set output \"$inputfile.eps\""		>> $gpfile
	
	echo  "plot	$alpha_gpu*x**$beta_gpu title \"GPU regression\" ,\\" >> $gpfile
	echo  "	\"$inputfile.gpu\" with errorbar title \"GPU measured\" ,\\" >> $gpfile
	echo  "	$c_gpu + exp(log($a_gpu) + $b_gpu * log(x) ) title \"GPU regression (non linear)\" ,\\" >> $gpfile
	echo  "	\"$inputfile.core\" with errorbar title \"CORE measured\" ,\\" >> $gpfile
	echo  "	$alpha_core*x**$beta_core title \"CORE regression\" ,\\" >> $gpfile
	echo  "	$c_core + exp(log($a_core) + $b_core * log(x) ) title \"CORE regression (non linear)\"" >> $gpfile
	
	gnuplot $gpfile
}

for file in $@
do
	trace_model "$file"
done
