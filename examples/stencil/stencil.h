/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#ifndef __STENCIL_H__
#define __STENCIL_H__

#define TYPE    float

#define DIM     (128)
#define BORDER  (1)

#define PADDING (64 / sizeof(TYPE) - 2*BORDER)
#define FIRST_PAD (PADDING+1)

#define REALDIM (DIM + 2 * BORDER + PADDING)

#define SURFACE   ((DIM + 2 * BORDER) * REALDIM)

#define SIZE    ((DIM + 2 * BORDER) * SURFACE + 1)

#define XBLOCK  (16)
#define YBLOCK  (16)
#define ZBLOCK  (64)

#define X_PER_THREAD (1)
#define Y_PER_THREAD (4)
#define Z_PER_THREAD (ZBLOCK)

#endif

