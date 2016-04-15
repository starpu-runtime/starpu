#!/bin/bash

SUPPRESSIONS=$(for f in $(dirname $0)/*.suppr ; do echo "--suppressions=$f" ; done)
valgrind $SUPPRESSIONS --leak-check=full --show-leak-kinds=all $*
