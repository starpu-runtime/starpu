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

#include <config.h>
#include <core/perfmodel/perfmodel.h>
#include "../helper.h"

int main(int argc, char **argv)
{
	int ret;

	ret = _starpu_check_number(42.0, 0);
	FPRINTF(stderr, "%s when reading %lf\n", ret?"Success":"Error", 42.0);

	if (ret)
	{
	     ret = _starpu_check_number(NAN, 1);
	     FPRINTF(stderr, "%s when reading %lf\n", ret?"Success":"Error", NAN);
	}

	return ret;
}
