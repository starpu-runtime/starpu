#!/bin/bash

root=$(dirname $0)
x=$(pwd | sed 's/.*_web_//')

if test -f $root/doxygen/chapters/starpu_$x/${x}_intro.doxy
then
    headline=$(grep -A2 intropage $root/doxygen/chapters/starpu_$x/${x}_intro.doxy | tail -1)
    echo "  - $x [[./starpu_web_$x.pdf][PDF]] - [[./html_web_$x/][HTML]] $headline">> $1/README
fi
