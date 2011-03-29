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
#include <stdarg.h>
#include <string.h>
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

/* Number of tasks submitted.  */
static unsigned int tasks_submitted;

struct insert_task_argument
{
  int     type;     /* `STARPU_VALUE', etc. */
  void   *pointer;  /* Pointer to the expected value.  */
  size_t  size;     /* Size in bytes of the data pointed to.  */
};

/* Pointer to a zero-terminated array listing the expected
   `starpu_insert_task' arguments.  */
const struct insert_task_argument *expected_insert_task_arguments;

void
starpu_insert_task (starpu_codelet *cl, ...)
{
  assert (cl->where == (STARPU_CPU | STARPU_OPENCL));

  /* TODO: Call `cpu_func' & co. and check whether they do the right
     thing.  */

  assert (cl->cpu_func != NULL);
  assert (cl->opencl_func != NULL);
  assert (cl->cuda_func == NULL);

  va_list args;

  va_start (args, cl);

  const struct insert_task_argument *expected;
  for (expected = expected_insert_task_arguments;
       expected->type != 0;
       expected++)
    {
      int type;
      void *arg;
      size_t size;

      type = va_arg (args, int);
      arg = va_arg (args, void *);
      size = va_arg (args, size_t);

      assert (type == expected->type);
      assert (size == expected->size);
      assert (arg != NULL);
      assert (!memcmp (arg, expected->pointer, size));
    }

  va_end (args);

  /* Make sure all the arguments were consumed.  */
  assert (expected->type == 0);

  tasks_submitted++;
}

/* Our own implementation of `starpu_unpack_cl_args', for debugging
   purposes.  */

void
starpu_unpack_cl_args (void *cl_raw_arg, ...)
{
  va_list args;
  size_t nargs, arg, offset, size;
  unsigned char *cl_arg;

  cl_arg = (unsigned char *) cl_raw_arg;

  nargs = *cl_arg;

  va_start (args, cl_raw_arg);

  for (arg = 0, offset = 1, size = 0;
       arg < nargs;
       arg++, offset += sizeof (size_t) + size)
    {
      void *argp;

      argp = va_arg (args, void *);
      size = *(size_t *) &cl_arg[size];

      memcpy (argp, &cl_arg[offset], size);
    }

  va_end (args);
}


int
main (int argc, char *argv[])
{
#pragma starpu hello

  int x = 42, y = 77;

  struct insert_task_argument expected[] =
    {
      { STARPU_VALUE, &x, sizeof (int) },
      { STARPU_VALUE, &y, sizeof (int) },
      { 0, 0, 0 }
    };

  expected_insert_task_arguments = expected;

  /* Invoke the task, which should make sure it gets called with
     EXPECTED.  */
  my_scalar_task (x, y);

  assert (tasks_submitted == 1);

  return EXIT_SUCCESS;
}
