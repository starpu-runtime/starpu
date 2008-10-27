#!/bin/bash

trace_model()
{
	inputfile=$1
	
	coreentries=`head -1 $inputfile`
	gpuentries=`head -2 $inputfile|tail -1`
	
	coremodel=`head -3 $inputfile|tail -1`
	gpumodel=`head -4 $inputfile|tail -1`
	
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
	echo "set title \"$inputfile\""			>> $gpfile 
	echo "set output \"$inputfile.eps\""		>> $gpfile
	
	echo  "plot	$alpha_gpu*x**$beta_gpu title \"GPU regression\" ,\\" >> $gpfile
	echo  "	$alpha_core*x**$beta_core title \"CORE regression\" ,\\" >> $gpfile
	echo  "	\"$inputfile.gpu\" with errorbar title \"GPU measured\" ,\\" >> $gpfile
	echo  "	\"$inputfile.core\" with errorbar title \"CORE measured\"" >> $gpfile
	
	gnuplot $gpfile
}

for file in $@
do
	trace_model "$file"
done
