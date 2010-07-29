/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
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
