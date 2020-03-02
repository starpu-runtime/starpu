/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __PXLU_KERNELS_H__
#define __PXLU_KERNELS_H__

#include <starpu.h>

#define str(s) #s
#define xstr(s)        str(s)
#define STARPU_PLU_STR(name)  xstr(STARPU_PLU(name))

extern struct starpu_codelet STARPU_PLU(cl11);
extern struct starpu_codelet STARPU_PLU(cl12);
extern struct starpu_codelet STARPU_PLU(cl21);
extern struct starpu_codelet STARPU_PLU(cl22);

#endif // __PXLU_KERNELS_H__
