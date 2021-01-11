/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

//! [To be included. You should update doxygen if you see this text.]
int main (void)
{
#pragma starpu initialize

#define NX     0x100000
#define FACTOR 3.14

  {
    float vector[NX]
       __attribute__ ((heap_allocated, registered));

    size_t i;
    for (i = 0; i < NX; i++)
      vector[i] = (float) i;

    vector_scal (NX, vector, FACTOR);

#pragma starpu wait
  } /* VECTOR is automatically freed here. */

#pragma starpu shutdown

  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
//! [To be included. You should update doxygen if you see this text.]
