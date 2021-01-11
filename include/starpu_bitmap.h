/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Simon Archipoff
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

#ifndef __STARPU_BITMAP_H__
#define __STARPU_BITMAP_H__

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_bitmap *starpu_bitmap_create(void) STARPU_ATTRIBUTE_MALLOC;
void starpu_bitmap_destroy(struct starpu_bitmap *b);

void starpu_bitmap_set(struct starpu_bitmap *b, int e);
void starpu_bitmap_unset(struct starpu_bitmap *b, int e);
void starpu_bitmap_unset_all(struct starpu_bitmap *b);

int starpu_bitmap_get(struct starpu_bitmap *b, int e);
void starpu_bitmap_unset_and(struct starpu_bitmap *a, struct starpu_bitmap *b, struct starpu_bitmap *c);
void starpu_bitmap_or(struct starpu_bitmap *a, struct starpu_bitmap *b);
int starpu_bitmap_and_get(struct starpu_bitmap *b1, struct starpu_bitmap *b2, int e);
int starpu_bitmap_cardinal(struct starpu_bitmap *b);

int starpu_bitmap_first(struct starpu_bitmap *b);
int starpu_bitmap_last(struct starpu_bitmap *b);
int starpu_bitmap_next(struct starpu_bitmap *b, int e);
int starpu_bitmap_has_next(struct starpu_bitmap *b, int e);

#ifdef __cplusplus
}
#endif

#endif
