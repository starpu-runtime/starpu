#!/bin/sh
OUTPUT=tasks_size_overhead.output
VALS=$(sed -n -e '4p' < $OUTPUT)
VAL1=$(echo "$VALS" | cut -d '	' -f 3)
VAL2=$(echo "$VALS" | cut -d '	' -f 5)
VAL3=$(echo "$VALS" | cut -d '	' -f 7)
VAL4=$(echo "$VALS" | cut -d '	' -f 9)
VAL5=$(echo "$VALS" | cut -d '	' -f 11)
VAL6=$(echo "$VALS" | cut -d '	' -f 13)
VAL7=$(echo "$VALS" | cut -d '	' -f 15)
gnuplot << EOF
set terminal eps
set output "tasks_size_overhead.eps"
set key top left
plot \
	"$OUTPUT" using 1:($VAL1)/(\$3) with linespoints title columnheader(2), \
	"$OUTPUT" using 1:($VAL2)/(\$5) with linespoints title columnheader(4), \
	"$OUTPUT" using 1:($VAL3)/(\$7) with linespoints title columnheader(6), \
	"$OUTPUT" using 1:($VAL4)/(\$9) with linespoints title columnheader(8), \
	"$OUTPUT" using 1:($VAL5)/(\$11) with linespoints title columnheader(10), \
	"$OUTPUT" using 1:($VAL6)/(\$13) with linespoints title columnheader(12), \
	"$OUTPUT" using 1:($VAL7)/(\$15) with linespoints title columnheader(14), \
	x
EOF
