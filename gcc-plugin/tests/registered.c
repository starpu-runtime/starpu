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

static void
test_vec (void)
{
  data_register_calls = data_unregister_calls = 0;
  expected_register_arguments.pointer = NULL;
  expected_register_arguments.elements = 123;
  expected_register_arguments.element_size = sizeof (float);

  float vec[123]		    /* FIXME: warning: "considered unsafe" */
    __attribute__ ((registered));

  assert (data_register_calls == 1);
  assert (expected_register_arguments.pointer == vec);

  expected_unregister_arguments.pointer = vec;
}

static void
test_matrix (void)
{
  data_register_calls = data_unregister_calls = 0;
  expected_register_arguments.pointer = NULL;
  expected_register_arguments.elements = 123;
  expected_register_arguments.element_size = 234 * sizeof (double);

  double matrix[123][234]	     /* FIXME: warning "considered unsafe" */
    __attribute__ ((registered));

  assert (data_register_calls == 1);
  assert (expected_register_arguments.pointer == matrix);

  expected_unregister_arguments.pointer = matrix;
}

static void
test_with_heap_alloc (void)
{
  data_register_calls = data_unregister_calls = 0;
  malloc_calls = free_calls = 0;

  expected_register_arguments.pointer = NULL;
  expected_register_arguments.elements = 123;
  expected_register_arguments.element_size =
     234 * 77 * sizeof (int);
  expected_malloc_argument =
    expected_register_arguments.elements
    * expected_register_arguments.element_size;

  int matrix[123][234][77]
    __attribute__ ((registered, heap_allocated));

  assert (data_register_calls == 1);
  assert (expected_register_arguments.pointer == matrix);
  assert (malloc_calls == 1);

  expected_unregister_arguments.pointer = matrix;
  expected_free_argument = matrix;
}

/* Same as above, but with the attributes in reverse order.  */
static void
test_with_heap_alloc_reversed (void)
{
  data_register_calls = data_unregister_calls = 0;
  malloc_calls = free_calls = 0;

  expected_register_arguments.pointer = NULL;
  expected_register_arguments.elements = 123;
  expected_register_arguments.element_size =
     234 * 77 * sizeof (int);
  expected_malloc_argument =
    expected_register_arguments.elements
    * expected_register_arguments.element_size;

  int matrix[123][234][77]
    __attribute__ ((heap_allocated, registered));

  assert (data_register_calls == 1);
  assert (expected_register_arguments.pointer == matrix);
  assert (malloc_calls == 1);

  expected_unregister_arguments.pointer = matrix;
  expected_free_argument = matrix;
}


int
main (int argc, char *argv[])
{
#pragma starpu initialize

  test_vec ();
  assert (data_unregister_calls == 1);

  test_matrix ();
  assert (data_unregister_calls == 1);

  test_with_heap_alloc ();
  assert (data_unregister_calls == 1);
  assert (free_calls == 1);

  test_with_heap_alloc_reversed ();
  assert (data_unregister_calls == 1);
  assert (free_calls == 1);

  return EXIT_SUCCESS;
}
