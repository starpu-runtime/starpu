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

/* Test error handling for the `task' and `task_implementation' attributes.  */

static void my_task (int foo, char *bar) __attribute__ ((task));
static void my_task_cpu (int foo, float *bar)    /* (error "type differs") */
  __attribute__ ((task_implementation ("cpu", my_task)));

static void my_task_opencl (long foo, char *bar) /* (error "type differs") */
  __attribute__ ((task_implementation ("opencl", my_task)));

static void my_task_nowhere (int foo, char *bar)
  __attribute__ ((task_implementation ("does-not-exist", my_task)));

static void my_task_not_quite (int foo, char *bar) /* (error "lacks the 'task' attribute") */
  __attribute__ ((task_implementation ("cpu", my_task_nowhere)));

static int foo /* (error "only applies to function") */
  __attribute__ ((task_implementation ("cpu", my_task)));

static int bar /* (error "only applies to function") */
  __attribute__ ((task, unused));

static int not_a_task __attribute__ ((unused));

static void my_task_almost (int foo, char *bar)    /* (error "not a function") */
  __attribute__ ((task_implementation ("cpu", not_a_task)));

static void my_task_wrong_task_arg (int foo, char *bar)   /* (error "not a function") */
  __attribute__ ((task_implementation ("cpu", 123)));

static void my_task_wrong_target_arg (int foo, char *bar) /* (error "string constant expected") */
  __attribute__ ((task_implementation (123, my_task)));


static void
my_task_cpu (int foo, float *bar)
{
}

static void
my_task_opencl (long foo, char *bar)
{
}

static void
my_task_nowhere (int foo, char *bar)  /* (warning "unsupported target") */
{
}

static void
my_task_not_quite (int foo, char *bar)
{
}

static void
my_task_almost (int foo, char *bar)
{
}

static void
my_task_wrong_task_arg (int foo, char *bar)
{
}

static void
my_task_wrong_target_arg (int foo, char *bar)
{
}
