/* GCC-StarPU
   Copyright (C) 2012 Institut National de Recherche en Informatique et Automatique

   GCC-StarPU is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GCC-StarPU is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GCC-StarPU.  If not, see <http://www.gnu.org/licenses/>.  */

#include <mocks.h>	    /* for `starpu_opencl_load_opencl_from_string' */

/* Claim that OpenCL is supported.  */
#pragma starpu add_target "opencl"


void my_task (int x, float a[x])
  __attribute__ ((task));

static void my_task_cpu (int x, float a[x])
  __attribute__ ((task_implementation ("cpu", my_task)));

static void my_task_opencl (int x, float a[x])
  __attribute__ ((task_implementation ("opencl", my_task)));

static void
my_task_cpu (int x, float a[x])
{
}


#pragma starpu opencl my_task "test.cl" "kern" /* (error "not a.* task impl") */
#pragma starpu opencl my_task_cpu  /* (error "not a.* task impl") */	\
                      "test.cl" "kern"
#pragma starpu opencl my_task_opencl "/dev/null" "kern" /* (error "empty") */
#pragma starpu opencl my_task_opencl "/does-not-exist/" "kern" /* (error "failed to access") */

#pragma starpu opencl my_task_opencl	  /* (error "wrong number of arg") */
#pragma starpu opencl my_task_opencl 123 "kern" /* (error "string constant") */
#pragma starpu opencl my_task_opencl "test.cl" 123 /* (error "string constant") */
#pragma starpu opencl my_task_opencl "test.cl" "kern" "foo" /* (error "junk after") */

void
foo (void)
{
#pragma starpu opencl my_task_opencl "test.cl" "kern" /* (error "top-level") */
}
