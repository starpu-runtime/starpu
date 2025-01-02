/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_SIMGRID_WRAP_H__
#define __STARPU_SIMGRID_WRAP_H__

#include <starpu_config.h>

#ifdef STARPU_SIMGRID
#ifndef main
#define main starpu_main
#ifdef __cplusplus
extern "C" int starpu_main(int argc, char *argv[]);
extern "C" int starpu_main(int argc, char **argv);
#endif
#endif
#endif

#endif /* __STARPU_SIMGRID_WRAP_H__ */
