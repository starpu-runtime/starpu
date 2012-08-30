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

#undef NDEBUG

#include <mocks.h>
#include <stdlib.h>

/* Claim that OpenCL is supported.  */
#pragma starpu add_target "opencl"


static void my_task (int x, float a[x])
  __attribute__ ((task));

static void my_task_opencl (int x, float a[x])
  __attribute__ ((task_implementation ("opencl", my_task)));

#pragma starpu opencl my_task_opencl "test.cl" "kern"

int
main ()
{
  static float a[123];

#pragma starpu initialize

  memset (a, 0, sizeof a);

  expected_register_arguments.pointer = a;
  expected_register_arguments.elements = sizeof a / sizeof a[0];
  expected_register_arguments.element_size = sizeof a[0];
#pragma starpu register a

  static int x = 123;
  struct insert_task_argument expected[] =
    {
      { STARPU_VALUE, &x, sizeof x },
      { STARPU_RW, a },
      { 0, 0, 0 }
    };

  expected_insert_task_arguments = expected;
  expected_insert_task_targets = STARPU_OPENCL;

  my_task (123, a);
  my_task (123, a);
  my_task (123, a);

  assert (tasks_submitted == 3);
  assert (load_opencl_calls == 1);

  return EXIT_SUCCESS;
}
