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

/* (instructions compile (cflags "-Wno-unused-variable")) */

static int global[123]              /* (error "cannot be used") */
  __attribute__ ((registered, used));

extern int external[123]            /* (error "cannot be used") */
  __attribute__ ((registered));

void
foo (size_t size)
{
  float scalar /* (error "must have an array type") */
    __attribute__ ((registered));
  float *ptr   /* (error "must have an array type") */
    __attribute__ ((registered));
  float incomp[]  /* (error "incomplete array type") */
    __attribute__ ((registered));
  float incomp2[size][3][]  /* (error "incomplete element type") */
    __attribute__ ((registered));
}
