/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016 Université de Bordeaux
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

/* Check that defining a main makes starpu use MSG_process_attach. */
#include "locality.c"
#include <config.h>
#if defined(HAVE_MSG_PROCESS_ATTACH) && SIMGRID_VERSION_MAJOR > 3 || (SIMGRID_VERSION_MAJOR == 3 && SIMGRID_VERSION_MINOR >= 14)
#undef main
int main(int argc, char *argv[]) {
	return starpu_main(argc, argv);
}
#endif
