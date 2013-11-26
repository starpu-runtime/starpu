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
/* Use the `task' attribute only when StarPU's GCC plug-in
   is available.   */
#ifdef STARPU_GCC_PLUGIN
# define __task  __attribute__ ((task))
#else
# define __task
#endif

static void matmul (const float *A, const float *B, float *C,
                    unsigned nx, unsigned ny, unsigned nz) __task;
//! [To be included]
