/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/**
   @defgroup API_Bitmap Bitmap
   @brief This is the interface for the bitmap utilities provided by StarPU.
   @{
 */

/** create a empty starpu_bitmap */
struct starpu_bitmap *starpu_bitmap_create(void) STARPU_ATTRIBUTE_MALLOC;
/** free \p b */
void starpu_bitmap_destroy(struct starpu_bitmap *b);

/** set bit \p e in \p b */
void starpu_bitmap_set(struct starpu_bitmap *b, int e);
/** unset bit \p e in \p b */
void starpu_bitmap_unset(struct starpu_bitmap *b, int e);
/** unset all bits in \p b */
void starpu_bitmap_unset_all(struct starpu_bitmap *b);

/** return true iff bit \p e is set in \p b */
int starpu_bitmap_get(struct starpu_bitmap *b, int e);
/** Basically compute \c starpu_bitmap_unset_all(\p a) ; \p a = \p b & \p c; */
void starpu_bitmap_unset_and(struct starpu_bitmap *a, struct starpu_bitmap *b, struct starpu_bitmap *c);
/** Basically compute \p a |= \p b */
void starpu_bitmap_or(struct starpu_bitmap *a, struct starpu_bitmap *b);
/** return 1 iff \p e is set in \p b1 AND \p e is set in \p b2 */
int starpu_bitmap_and_get(struct starpu_bitmap *b1, struct starpu_bitmap *b2, int e);
/** return the number of set bits in \p b */
int starpu_bitmap_cardinal(struct starpu_bitmap *b);

/** return the index of the first set bit of \p b, -1 if none */
int starpu_bitmap_first(struct starpu_bitmap *b);
/** return the position of the last set bit of \p b, -1 if none */
int starpu_bitmap_last(struct starpu_bitmap *b);
/** return the position of set bit right after \p e in \p b, -1 if none */
int starpu_bitmap_next(struct starpu_bitmap *b, int e);
/** todo */
int starpu_bitmap_has_next(struct starpu_bitmap *b, int e);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
