#!/bin/bash

stcolor=$(tput sgr0)
datacolor=$(tput setaf 2)
filecolor=$(tput setaf 1)

process_file()
{
    datas=$(grep "data_register(" $f| awk -F',' '{print $1}' | awk -F'(' '{print $2}' | tr -d '&' | sed 's/\[/\\\[/g' | sed 's/\]/\\\]/g' | sed 's/\*/\\\*/g')
    for data in $datas ; do
	x=$(grep "data_unregister($data" $1)
	if test "$x" == "" ; then
	    x=$(grep "data_unregister_no_coherency($data" $1)
	    if test "$x" == "" ; then
		echo "Error. File <${filecolor}$1${stcolor}>. Handle <${datacolor}$data${stcolor}> is not unregistered"
	    fi
	fi
    done
}

for f in $* ; do process_file $f ; done
