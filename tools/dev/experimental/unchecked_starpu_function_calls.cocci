/*
 * StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 inria
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
 * The return values of functions such as starpu_init(), starpu_task_submit(),
 * starpu_task_wait() should _always_ be checked. This semantic patch looks for
 * calls to starpu_task_submit() where the return value is ignored. It could
 * probably be extended to apply to other functions as well.
 */
virtual context
virtual org
virtual patch
virtual report

@initialize:python depends on report || org@
msg = "Unchecked call to starpu_task_submit()"

@unchecked_starpu_func_call@
identifier f;
position p;
@@
f(...)
{
	...
	starpu_task_submit@p(...);
	...
}


// Context mode.
@depends on unchecked_starpu_func_call && context@
position unchecked_starpu_func_call.p;
@@
* starpu_task_submit@p(...);

// Org mode.
@script:python depends on unchecked_starpu_func_call && org@
p << unchecked_starpu_func_call.p;
@@
coccilib.org.print_todo(p[0], msg)

// Patch mode.
@has_ret depends on unchecked_starpu_func_call@
identifier unchecked_starpu_func_call.f;
identifier ret;
identifier starpu_func =~ "^starpu_";
@@
f(...)
{
	...
	ret = starpu_func(...);
	...
}

@depends on unchecked_starpu_func_call && has_ret && patch@
identifier unchecked_starpu_func_call.f;
identifier has_ret.ret;
@@
f(...)
{
...
- starpu_task_submit(
+ ret = starpu_task_submit(
...);
+ STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
...
}

@depends on unchecked_starpu_func_call && !has_ret && patch@
identifier unchecked_starpu_func_call.f;
@@
f(...)
{
...
- starpu_task_submit(
+ int ret = starpu_task_submit(
...);
+ STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

...
}

// Report mode.
@script:python depends on unchecked_starpu_func_call && report@
p << unchecked_starpu_func_call.p;
@@
coccilib.report.print_report(p[0], msg)
