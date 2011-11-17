#!/bin/bash
# Note: expects Coccinelle's spatch command n the PATH
# See: http://coccinelle.lip6.fr/
stcolor=$(tput sgr0)
redcolor=$(tput setaf 1)

handles=$(spatch -very_quiet -sp_file tools/dev/starpu_check_register.cocci "$@")
if test "x$handles" != "x" ; then
	for handle in $handles; do
		echo "$handle"
		register=$(echo $handle|awk -F ',' '{print $1}')
		location=$(echo $handle|awk -F ',' '{print $2}')
		echo "data handle ${redcolor}${register}${stcolor} registered at location $location does not seem to be properly unregistered"
	done
fi
