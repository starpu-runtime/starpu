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

#include <mocks.h>

static void
foo (size_t size)
{
  expected_malloc_argument = size * 23 * sizeof (float);

  float m[size][23] __attribute__ ((heap_allocated));

  assert (malloc_calls == 1);

  /* Freed when going out of scope.  */
  expected_free_argument = m;
}

int
main (int argc, char *argv[])
{
#pragma starpu initialize

  foo (42 + argc);

  assert (free_calls == 1);

  return EXIT_SUCCESS;
}
