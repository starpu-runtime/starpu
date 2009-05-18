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

# we want to handle requests like *.debug
for inputfile in "$@"
do
echo "Handle file $inputfile"
hashlist=`cut -f 1 $inputfile | sort | uniq | xargs` 

# extract subfiles from the history file
for h in $hashlist
do
	echo "Handling tasks with hash = $h"
	grep "^$h" $inputfile| cut -f 2- > $inputfile.$h

R --no-save > /dev/null << EOF

table <- read.table("$inputfile.$h")

x <- table[,1]
hist(x[x > quantile(x,0.01) & x<quantile(x,0.99)], col="red", breaks=50, density=10)

EOF
mv Rplots.pdf $inputfile.$h.pdf
done

done
echo "finished !"
