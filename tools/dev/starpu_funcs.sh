#!/bin/bash
# Note: expects Coccinelle's spatch command n the PATH
# See: http://coccinelle.lip6.fr/

stcolor=$(tput sgr0)
redcolor=$(tput setaf 1)
greencolor=$(tput setaf 2)

functions=$(spatch -sp_file tools/dev/starpu_funcs.cocci $(find include -name '*.h'))
for func in $functions ; do
	fname=$(echo $func|awk -F ',' '{print $1}')
	location=$(echo $func|awk -F ',' '{print $2}')
	x=$(grep $fname doc/starpu.texi | grep deftypefun)
	if test "$x" == "" ; then
		echo "function ${redcolor}${fname}${stcolor} at location $location is not (or incorrectly) documented"
	else
		echo "function ${greencolor}${fname}${stcolor} at location $location is correctly documented"
	fi

done
