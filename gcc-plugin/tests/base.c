/* GCC-StarPU
   Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique

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

#undef NDEBUG

#include <stdlib.h>
#include <assert.h>

#ifndef STARPU_GCC_PLUGIN
# error barf!
#endif

#ifndef STARPU_USE_CPU
# error damn it!
#endif


/* The task under test.  */

static void my_scalar_task (int x, int y) __attribute__ ((task));

static void my_scalar_task_cpu (int, int)
  __attribute__ ((task_implementation ("cpu", my_scalar_task)));
static void my_scalar_task_opencl (int, int)
  __attribute__ ((task_implementation ("opencl", my_scalar_task)));

static void
my_scalar_task_cpu (int x, int y)
{
  printf ("%s: x = %i, y = %i\n", __func__, x, y);
}

static void
my_scalar_task_opencl (int x, int y)
{
  printf ("%s: x = %i, y = %i\n", __func__, x, y);
}


/* Stub used for testing purposes.  */

static unsigned int tasks_submitted;

void
starpu_insert_task (starpu_codelet *cl, ...)
{
  assert (cl->where == (STARPU_CPU | STARPU_OPENCL));

  /* XXX: Eventually the `_func' field will point to a wrapper instead of
     pointing directly to the task implementation.  */
  assert (cl->cpu_func == (void *) my_scalar_task_cpu);
  assert (cl->opencl_func == (void *) my_scalar_task_opencl);
  assert (cl->cuda_func == NULL);

  tasks_submitted++;
}


int
main (int argc, char *argv[])
{
#pragma starpu hello

  int x = 42, y = 77;

  my_scalar_task (x, y);
  assert (tasks_submitted == 1);

  return EXIT_SUCCESS;
}
