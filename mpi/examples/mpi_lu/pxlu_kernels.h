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

#ifndef __PXLU_KERNELS_H__
#define __PXLU_KERNELS_H__

#include <starpu.h>

#define str(s) #s
#define xstr(s)        str(s)
#define STARPU_PLU_STR(name)  xstr(STARPU_PLU(name))

starpu_codelet STARPU_PLU(cl11);
starpu_codelet STARPU_PLU(cl12);
starpu_codelet STARPU_PLU(cl21);
starpu_codelet STARPU_PLU(cl22);

#endif // __PXLU_KERNELS_H__
