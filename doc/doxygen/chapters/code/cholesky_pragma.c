/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

//! [To be included]
extern void cholesky(unsigned nblocks, unsigned size,
                    float mat[nblocks][nblocks][size])
  __attribute__ ((task));

int
main (int argc, char *argv[])
{
#pragma starpu initialize

  /* ... */

  int nblocks, size;
  parse_args (&nblocks, &size);

  /* Allocate an array of the required size on the heap,
     and register it.  */

  {
    float matrix[nblocks][nblocks][size]
      __attribute__ ((heap_allocated, registered));

    cholesky (nblocks, size, matrix);

#pragma starpu wait

  }   /* MATRIX is automatically unregistered & freed here.  */

#pragma starpu shutdown

  return EXIT_SUCCESS;
}
//! [To be included]
