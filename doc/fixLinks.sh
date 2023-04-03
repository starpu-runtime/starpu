#!/bin/bash
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

root=$(dirname $0)
root_src=$root
root_build=$1

files=$(find $root_build -name "*html")
if test "$files" == ""
then
    # there is no html files to process
    exit
fi

for d in $root_src/doxygen/chapters/starpu_*
do
    for f in $(find $d -name "*.doxy")
    do
	#echo $f
	part=$(basename $(dirname $f))
	link=$(grep -F "\page" $f | awk '{print $3}')
	if test -z "$link"
	then
	    continue
	fi

	x1=$(echo $part | sed 's/starpu/doxygen_web/')
	x2=$(echo $part | sed 's/starpu/html_web/')
	title=$(grep -F "\page" $f | sed 's;..! .page '$link';;')
	#echo $part
	#echo $link
	#echo $f
	#echo $title

	# we replace the link with the correct link in the installation directory, it will not work in the build directory
	# there we would have to use ../../$x1/$x2/${link}.html
	for ff in $(grep -lrs "Chapter $link" $(find $root_build -name "*html"))
	do
	    script=$(mktemp)
	    echo "sed -i 's;Chapter "$link";Chapter <a class=\"el\" href=\"../"$x2"/"${link}".html\">"$title"</a>;' $ff" > $script
	    . $script
	done
    done
done
