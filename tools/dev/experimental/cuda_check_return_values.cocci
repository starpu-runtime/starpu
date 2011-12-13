// StarPU --- Runtime system for heterogeneous multicore architectures.
//
// Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique
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
@initialize:python@
l = []


@seek@
identifier func;
expression E;
statement S1, S2;
position p;
identifier cuda_func =~ "^cuda";
@@
func(...)
{
...
E@p = cuda_func(...);
... when != if (!E) S1
    when != if (!E) S1 else S2
    when != if (E) S1
    when != !E
    when != E != cudaSuccess
    when != E == cudaSuccess
    when != STARPU_UNLIKELY(!E)
}

@fix@
expression seek.E;
position seek.p;
identifier seek.cuda_func;
@@
E@p = cuda_func(...);
+ if (STARPU_UNLIKELY(E != cudaSuccess))
+	STARPU_CUDA_REPORT_ERROR(E);


@no_assignment@
identifier cuda_func =~ "^cuda";
position p;
@@
cuda_func@p(...);


@script:python@
p << no_assignment.p;
func << no_assignment.cuda_func;
@@
l.append((p[0], func));


@finalize:python@
for e in l:
	p, f = e[0], e[1]
	print "%s:%s : the return value of %s is ignored" % (p.file, p.line, f)
print "This should probably be patched by either :"
print " * Checking the return value of the function : cudaError_t ret = f(...);"
print " * Explicitely ignoring its return value : (void) f(...)"

