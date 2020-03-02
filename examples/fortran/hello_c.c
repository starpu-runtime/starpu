/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This shows how to call a fortran function from a C function
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <f77.h>

#define fline_length 80

extern F77_SUBROUTINE(hellosub)( INTEGER(i) TRAIL(line) );


void dummy_c_func_(INTEGER(i))
{
	fprintf(stderr, "i = %d\n", *INTEGER_ARG(i));

	F77_CALL(hellosub)(INTEGER_ARG(i)TRAIL_ARG(fline));
}
