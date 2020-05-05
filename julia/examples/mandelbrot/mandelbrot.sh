#!/bin/bash

$(dirname $0)/../execute.sh mandelbrot/mandelbrot.jl
$(dirname $0)/../execute.sh mandelbrot/mandelbrot_native.jl
$(dirname $0)/../execute.sh -calllib mandelbrot/cpu_mandelbrot.c mandelbrot/mandelbrot.jl

