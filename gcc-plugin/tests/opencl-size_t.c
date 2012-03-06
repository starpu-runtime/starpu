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

/* Make sure use of `size_t' as a task argument type is flagged.  */

#undef NDEBUG

#include <mocks.h>
#include <unistd.h>

static void my_task (size_t size, int x[size]) __attribute__ ((task));

static void my_task_cpu (size_t size, int x[size])
  __attribute__ ((task_implementation ("cpu", my_task)));
static void my_task_opencl (size_t size, int x[size]) /* (error "not a valid OpenCL type") */
  __attribute__ ((task_implementation ("opencl", my_task)));

static void
my_task_cpu (size_t size, int x[size])
{
}

static void
my_task_opencl (size_t size, int x[size])
{
}
