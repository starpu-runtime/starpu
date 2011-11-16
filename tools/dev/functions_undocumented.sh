#!/bin/bash

stcolor=$(tput sgr0)
redcolor=$(tput setaf 1)

functions=$(grep -rs 'starpu.*(.*);' include | awk -F':' '{print $2}' | sed 's/(.*//' | sed 's/.* //'| tr -d ' ' | tr -d '*')

for func in $functions ; do
    #echo Processing function $func
    x=$(grep $func doc/starpu.texi | grep deftypefun)
    if test "$x" == "" ; then
        echo "Error. Function ${redcolor}${func}${stcolor} is not (or incorrectly) documented"
    fi
done


