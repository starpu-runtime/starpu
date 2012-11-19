// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2012 inria
//
// StarPU is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.
//
// StarPU is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// See the GNU Lesser General Public License in COPYING.LGPL for more details.

// OpenCL and CUDA functions are not very likely to fail, so we probably want
// to use branch predictions when checking their return value. This Coccinelle
// script tries to find places where this can be done.

virtual context
virtual org
virtual patch
virtual report

@r@
identifier ret;
statement S;
position p;
@@
if@p(
(
ret != CL_SUCCESS
|
ret != cudaSuccess
)
 ) S

@depends on context@
position r.p;
statement r.S;
@@
* if @p(...)
S

@script:python depends on org@
p << r.p;
@@
coccilib.org.print_todo(p[0], "Use STARPU_UNLIKELY")

@depends on patch@
position r.p;
expression E;
statement r.S;
@@
- if@p(E)
+ if (STARPU_UNLIKELY(E))
S

@script:python depends on report@
p << r.p;
@@
coccilib.report.print_report(p[0], "Use STARPU_UNLIKELY")
