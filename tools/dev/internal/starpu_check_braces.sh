#!/bin/sh

for d in tools src tests examples
do
    for ext in c h cl cu
    do
	grep -rsn "{" $d |grep ".${ext}:" | grep -v "}" | grep -v ".${ext}:[0-9]*:[[:space:]]*{$" > /tmp/braces
	if test -s /tmp/braces
	then
	    less /tmp/braces
	fi
	grep -rsn "}" $d |grep ".${ext}:" | grep -v "{" | grep -v "};" | grep -v ".${ext}:[0-9]*:[[:space:]]*};*$" > /tmp/braces
	if test -s /tmp/braces
	then
	    less /tmp/braces
	fi
    done
done
