#!/bin/bash

dirname=$(dirname $0)

x=$(grep ingroup $dirname/../chapters/api/*.doxy $dirname/../chapters/api/sc_hypervisor/*.doxy |awk -F':' '{print $2}'| awk 'NF != 2')
if test -n "$x" ; then
    echo Errors on group definitions
    echo $x
fi

echo
echo "Defined groups"
grep ingroup $dirname/../chapters/api/*.doxy $dirname/../chapters/api/sc_hypervisor/*.doxy|awk -F':' '{print $2}'| awk 'NF == 2'|sort|uniq
echo

for f in $dirname/../../../build/doc/doxygen/latex/*tex ; do
    x=$(grep $(basename $f .tex) $dirname/../refman.tex)
    if test -z "$x" ; then
	echo Error. $f not included in refman.tex
    fi
done

