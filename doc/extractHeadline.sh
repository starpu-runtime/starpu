#!/bin/bash

root=$(dirname $0)

(
    echo "** Full documentation"
    echo "  - [[./starpu.pdf][PDF]] - [[./html/][HTML]]"
    echo "** Parts of the documentation"
    for doc in doxygen_web_introduction doxygen_web_installation doxygen_web_basics doxygen_web_applications doxygen_web_performances doxygen_web_faq doxygen_web_languages doxygen_web_extensions
    do
	x=$(echo $doc | sed 's/.*_web_//')

	if test -f $root/doxygen/chapters/starpu_$x/${x}_intro.doxy
	then
	    headline=$(grep -A2 intropage $root/doxygen/chapters/starpu_$x/${x}_intro.doxy | tail -1)
	    echo "- $x"
	    if test -n "$headline"
	    then
		echo "  - $headline"
	    fi
	    echo "  - [[./starpu_web_$x.pdf][PDF]] - [[./html_web_$x/][HTML]]"
	fi
    done
    echo "** Developers documentation"
    echo "  - [[./starpu_dev.pdf][PDF]] - [[./html_dev/][HTML]]"
) > ./README.org
