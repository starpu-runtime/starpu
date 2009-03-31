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


# 
# # C files and C headers
# cheader=./scripts/license/cheader
# cfiles=`find . -name "*.[ch]"`
# 
# for filename in $cfiles
# do
# 	echo "licensing file $filename"
# 	
# 	cp $cheader $filename.tmp
# 	cat $filename >> $filename.tmp
# 	mv $filename.tmp $filename
# done
# 
# 
# # cuda files
# cheader=./scripts/license/cheader
# cfiles=`find . -name "*.cu"`
# 
# for filename in $cfiles
# do
# 	echo "licensing file $filename"
# 	
# 	cp $cheader $filename.tmp
# 	cat $filename >> $filename.tmp
# 	mv $filename.tmp $filename
# done
# 
# # Script & Makefiles
# mheader=./scripts/license/sheader
# mfiles=`find . -name "Makefile" -o -name "Makefile.in"`
# 
# for filename in $mfiles
# do
# 	echo "licensing file $filename"
# 	
# 	cp $mheader $filename.tmp
# 	cat $filename >> $filename.tmp
# 	mv $filename.tmp $filename
# done
# 

# Script & Makefiles
# rheader=./scripts/license/sheader
# rfiles=`find . -name "*.r"`
# 
# for filename in $rfiles
# do
# 	echo "licensing file $filename"
# 	
# 	cp $rheader $filename.tmp
# 	cat $filename >> $filename.tmp
# #	mv $filename.tmp $filename
# done
# 
# 


# sheader=./scripts/license/sheader
# sfiles=`find . -name "*.sh"`
# 
# for filename in $sfiles
# do
# 	echo "licensing file $filename"
# 
# 	length=`wc -l $filename | cut -f 1 -d ' '`
# 	
# 	head -1 $filename > $filename.tmp
# 	echo "" >> $filename.tmp
# 	cat $sheader >> $filename.tmp
# 	tail -$((length - 1)) $filename >> $filename.tmp
# 	mv $filename.tmp $filename
# done
# 
# 

sheader=./scripts/license/sheader
sfiles=`find . -name "*.gp"`

for filename in $sfiles
do
	echo "licensing file $filename"

	length=`wc -l $filename | cut -f 1 -d ' '`
	
	head -1 $filename > $filename.tmp
	echo "" >> $filename.tmp
	cat $sheader >> $filename.tmp
	tail -$((length - 1)) $filename >> $filename.tmp
	mv $filename.tmp $filename
done



# 
# 
# # fortran files
# fheader=./scripts/license/fheader
# cfiles=`find . -name "*.F"`
# 
# for filename in $cfiles
# do
# 	echo "licensing file $filename"
# 	
# 	cp $fheader $filename.tmp
# 	cat $filename >> $filename.tmp
# 	mv $filename.tmp $filename
# done
# 
