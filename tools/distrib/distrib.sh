#!/bin/bash

filename=$1

symbol_list=`cut -f1 $filename|sort -u`

for symbol in $symbol_list
do
	echo $symbol

	grep "^$symbol" $filename > output.$symbol
	
	
done
