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

virtual context
virtual org
virtual patch
virtual report

@r@
identifier f;
position p;
@@
f@p()
{
...
}

@depends on r && context@
identifier r.f;
position r.p;
@@
* f@p()
{
...
}

@script:python depends on r && org@
f << r.f;
p << r.p;
@@
coccilib.org.print_todo(p[0], "Fix non-ANSI function declaration of function %s." % f)

@depends on r && patch@
identifier r.f;
position r.p;
@@
-f@p()
+f(void)
{
...
}

@script:python depends on r && report@
f << r.f;
p << r.p;
@@
coccilib.report.print_report(p[0], "Non-ANSI function declaration of function %s." % f)
