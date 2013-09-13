/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Centre National de la Recherche Scientifique
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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <config.h>

int _starpu_read_double(FILE *f, char *format, double *val)
{
#ifdef STARPU_HAVE_WINDOWS
/** Windows cannot read NAN values, yes, it is really bad ... */
	int x1 = getc(f);
	int x2 = getc(f);
	int x3 = getc(f);

	if (x1 == 'n' && x2 == 'a' && x3 == 'n')
	{
		*val = NAN;
		return 1;
	}
	else
	{
		ungetc(x3, f);
		ungetc(x2, f);
		ungetc(x1, f);
		return fscanf(f, format, val);
	}
#else
	return fscanf(f, format, val);
#endif
}

#define STRING "booh"

int _starpu_check_number(double val, int nan)
{
	char *filename = tmpnam(NULL);

	/* write the double value in the file followed by a predefined string */
	FILE *f = fopen(filename, "w");
	fprintf(f, "%lf %s\n", val, STRING);
	fclose(f);

	/* read the double value and the string back from the file */
	f = fopen(filename, "r");
	double lat;
	char str[10];
	int x = _starpu_read_double(f, "%lf", &lat);
	int y = fscanf(f, "%s", str);
	fclose(f);

	/* check that what has been read is identical to what has been written */
	int pass;
	pass = (x == 1) && (y == 1);
	pass = pass && strcmp(str, STRING) == 0;
	if (nan)
		pass = pass && isnan(val) && isnan(lat);
	else
		pass = pass && lat == val;
	return pass;
}
