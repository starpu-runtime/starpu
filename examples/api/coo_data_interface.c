/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

// This program checks that the implementation of the COO data
// interface only uses StarPU's public API

#define starpu_interface_coo_ops my_starpu_interface_coo_ops
#define starpu_coo_data_register my_starpu_coo_data_register
#include "../../src/datawizard/interfaces/coo_interface.c"

int main()
{
	return 0;
}
