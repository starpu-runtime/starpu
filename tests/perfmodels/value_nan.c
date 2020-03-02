/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <common/config.h>
#include <core/perfmodel/perfmodel.h>
#include "../helper.h"
#include <unistd.h>

#ifdef STARPU_HAVE_WINDOWS
#include <io.h>
#include <fcntl.h>
#endif

/*
 * Test that _starpu_write_double and _starpu_read_double properly manipulate
 * NaN values
 */

#define STRING "booh"

static
int _check_number(double val, int checknan)
{
	char *tmp = "starpu_XXXXXX";
	char filename[100];

	strcpy(filename, tmp);

#ifdef STARPU_HAVE_WINDOWS
        _mktemp(filename);
#else
	{
	     int id = mkstemp(filename);
	     /* fail */
	     if (id < 0)
	     {
		  FPRINTF(stderr, "Error when creating temp file\n");
		  return 1;
	     }
	}
#endif

	/* write the double value in the file followed by a predefined string */
	FILE *f = fopen(filename, "w");
	if (!f)
	{
		FPRINTF(stderr, "Error when opening file %s\n", filename);
		return 1;
	}
	// A double is written with the format %e ...
	_starpu_write_double(f, "%e", val);
	fprintf(f, " %s\n", STRING);
	fclose(f);

	/* read the double value and the string back from the file */
	f = fopen(filename, "r");
	if (!f)
	{
		FPRINTF(stderr, "Error when opening file %s\n", filename);
		return 1;
	}
	double lat;
	char str[10];
	// ... but is read with the format %le
	int x = _starpu_read_double(f, "%le", &lat);
	int y = fscanf(f, " %9s", str);
	fclose(f);
	unlink(filename);

	/* check that what has been read is identical to what has been written */
	int pass;
	pass = (x == 1) && (y == 1);
	pass = pass && strcmp(str, STRING) == 0;
	if (checknan)
		pass = pass && isnan(val) && isnan(lat);
	else
		pass = pass && (int)lat == (int)val;
	return pass?0:1;
}

int main(void)
{
	int ret1, ret2;
	double nanvalue = nan("");

	ret1 = _check_number(42.0, 0);
	FPRINTF(stderr, "%s when reading %e\n", ret1==0?"Success":"Error", 42.0);

	ret2 = _check_number(nanvalue, 1);
	FPRINTF(stderr, "%s when reading %e\n", ret2==0?"Success":"Error", nanvalue);

	return ret1+ret2;
}
