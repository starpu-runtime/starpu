# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
audio
	This applies a simple band filter over audio files

axpy
	This computes the AXPY BLAS over a big vector

basic_examples
        This contains very trivial examples: hello world, scaling a vector, etc.

binary
	This shows how to store and load compiled OpenCL kernels on and from the
	file system

callback
	This shows how to use task callbacks

cg
	This computes a Conjugate Gradient

cholesky
	This computes a Cholesky factorization

common
	This holds common code for BLAS kernels

cpp
	This shows how to use StarPU from C++

filters
	This contains several partitioning examples

fortran90
	This shows how to use StarPU from Fortran90

gl_interop
	This shows how interoperation can be done between StarPU CUDA
	computations and OpenGL rendering

heat
        This uses a finite element method to compute heat propagation thanks to
        an LU factorization or a conjugate gradient

incrementer
	This just increments a variable

interface
        This shows how to implement a user-defined data type, here simply
        complex floats

lu
	This computes an LU factorization

mandelbrot
	This computes and outputs the mandelbrot set

matvecmult
	This computes a matrix-vector multiplication

mult
	This computes a matrix-matrix multiplication

openmp
	This shows how to use an OpenMP code inside a StarPU parallel task

pi
	This computes Pi thanks to random numbers

pipeline
	This shows how to submit a pipeline to StarPU with limited buffer
	use, and avoiding submitted all the tasks at once

ppm_downscaler
	This downscales PPM pictures

profiling
        This examplifies how to get profiling information on executed tasks

reductions
	This examplifies how to use value reductions

sched_ctx
	This examplifies how to use scheduling contexts

sched_ctx_utils
	This is just common code for scheduling contexts

scheduler
	This examplifies how to implement a user-defined scheduler

spmd
	This shows how to define a parallel task

spmv
	This computes a sparse matrix-vector multiplication

stencil
	This computes a dumb 3D stencil with 1D subdomain decomposition

tag_example
	This examplifies how to use tags for dependencies

top
	This examplifies how to enrich StarPU-top with information

worker_collections
	This examplifies how to use worker collections
