/*
 * StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 INRIA
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/*
 * It is a bad idea to write code such as :
 *
 * for (i = 0; i < foo(...); i++) { ... }
 *
 * Indeed, foo will be called every time we enter the loop. This would be better :
 *
 * unsigned int max = foo(...);
 * for (i = 0; i < max; i++) { ... }
 *
 * This semantic patch does not automagically generate a patch, but still
 * points out that kind of code so that it can be fixed (if necesary) by
 * programmers.
 */

/*
 * You may want to run spatch(1) with either -D report or -D org.
 * Otherwise, a context output will be generated.
 */
virtual report
virtual org 

@initialize:python depends on report || org@
msg="Function call in the termination condition of a for loop"

@r@
identifier f;
identifier it; 
expression E;
position p;
@@
for (it = E; it < f@p(...); ...)
{
...
}

@script:python depends on r && report@
p << r.p;
@@
coccilib.report.print_report(p[0], msg)

@script:python depends on r && org@
p << r.p;
@@
msg="Function call in the termination condition of a for loop"
coccilib.org.print_todo(p[0], msg)

@depends on !report && !org && r@
identifier r.f;
identifier r.it;
expression r.E;
@@
* for (it = E; it < f(...); ...)
{
...
}
