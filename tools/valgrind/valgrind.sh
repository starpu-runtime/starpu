#!/bin/bash

EXEC=$(basename $0 .sh)
if test "$EXEC" == "valgrind"
then
    RUN="valgrind"
else
    RUN="valgrind --tool=$EXEC"
fi
SUPPRESSIONS=$(for f in $(dirname $0)/*.suppr ; do echo "--suppressions=$f" ; done)
$RUN -v --num-callers=42 --error-exitcode=42 --track-origins=yes --leak-check=full --show-reachable=yes --errors-for-leak-kinds=all --show-leak-kinds=all --gen-suppressions=all $SUPPRESSIONS $*
