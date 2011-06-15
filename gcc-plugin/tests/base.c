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

#include <lib.h>


/* The task under test.  */

static void my_scalar_task (int x, int y, int z) __attribute__ ((task));

static void my_scalar_task_cpu (int, int, int)
  __attribute__ ((task_implementation ("cpu", my_scalar_task)));
static void my_scalar_task_opencl (int, int, int)
  __attribute__ ((task_implementation ("opencl", my_scalar_task)));

static void
my_scalar_task_cpu (int x, int y, int z)
{
  printf ("%s: x = %i, y = %i, z = %i\n", __func__, x, y, z);
}

static void
my_scalar_task_opencl (int x, int y, int z)
{
  printf ("%s: x = %i, y = %i, z = %i\n", __func__, x, y, z);
}


int
main (int argc, char *argv[])
{
#pragma starpu hello

  int x = 42, y = 77, z = 99;

  struct insert_task_argument expected[] =
    {
      { STARPU_VALUE, &x, sizeof (int) },
      { STARPU_VALUE, &y, sizeof (int) },
      { STARPU_VALUE, &z, sizeof (int) },
      { 0, 0, 0 }
    };

  expected_insert_task_arguments = expected;

  /* Invoke the task, which should make sure it gets called with
     EXPECTED.  */
  my_scalar_task (x, y, z);

  /* Invoke the task using literal constants instead of variables.  */
  my_scalar_task (42, y, z);
  my_scalar_task (x, 77, z);
  my_scalar_task (x, y, 99);

  my_scalar_task (42, 77, z);
  my_scalar_task (x, 77, 99);
  my_scalar_task (42, y, 99);

  my_scalar_task (42, 77, 99);

  assert (tasks_submitted == 8);

  return EXIT_SUCCESS;
}
