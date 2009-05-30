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

create_histograms()
{

inputfile=$1

R --no-save > /dev/null << EOF

handle_hash <- function (hash)
{

val <- table[table[,1]==hash,3]


# there is certainly a better way to do this !
size <- unique(table[table[,1]==hash,2])

pdf(paste("$inputfile", hash, size, "pdf", sep="."));

h <- hist(val[val > quantile(val,0.01) & val<quantile(val,0.99)], col="red", breaks=50, density=10)


}

table <- read.table("$inputfile")

hashlist <- unique(table[,1])

for (hash in hashlist)
{
	print(hash)
	handle_hash(hash)
}

EOF

}

for inputfile in $@
do
	create_histograms $inputfile 
done
