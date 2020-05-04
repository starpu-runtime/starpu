#!/bin/bash

$(dirname $0)/../execute.sh vector_scal/vector_scal.jl
$(dirname $0)/../execute.sh -calllib vector_scal/cpu_vector_scal.c vector_scal/vector_scal.jl

