#!/bin/bash

$(dirname $0)/../execute.sh mult/mult.jl
$(dirname $0)/../execute.sh mult/mult_native.jl
$(dirname $0)/../execute.sh -calllib mult/cpu_mult.c mult/mult.jl


