/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_util.h>
#include <starpu_config.h>

#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup API_Bitmap Bitmap
   @brief This is the interface for the bitmap utilities provided by StarPU.
   @{
*/
#ifndef _STARPU_LONG_BIT
#define _STARPU_LONG_BIT ((int)(sizeof(unsigned long) * 8))
#endif

#define _STARPU_BITMAP_SIZE ((STARPU_NMAXWORKERS - 1) / _STARPU_LONG_BIT) + 1

/** create a empty starpu_bitmap */
static inline struct starpu_bitmap *starpu_bitmap_create(void) STARPU_ATTRIBUTE_MALLOC;
/** zero a starpu_bitmap */
static inline void starpu_bitmap_init(struct starpu_bitmap *b);
/** free \p b */
static inline void starpu_bitmap_destroy(struct starpu_bitmap *b);

/** set bit \p e in \p b */
static inline void starpu_bitmap_set(struct starpu_bitmap *b, int e);
/** unset bit \p e in \p b */
static inline void starpu_bitmap_unset(struct starpu_bitmap *b, int e);
/** unset all bits in \p b */
static inline void starpu_bitmap_unset_all(struct starpu_bitmap *b);

/** return true iff bit \p e is set in \p b */
static inline int starpu_bitmap_get(struct starpu_bitmap *b, int e);
/** Basically compute \c starpu_bitmap_unset_all(\p a) ; \p a = \p b & \p c; */
static inline void starpu_bitmap_unset_and(struct starpu_bitmap *a, struct starpu_bitmap *b, struct starpu_bitmap *c);
/** Basically compute \p a |= \p b */
static inline void starpu_bitmap_or(struct starpu_bitmap *a, struct starpu_bitmap *b);
/** return 1 iff \p e is set in \p b1 AND \p e is set in \p b2 */
static inline int starpu_bitmap_and_get(struct starpu_bitmap *b1, struct starpu_bitmap *b2, int e);
/** return the number of set bits in \p b */
static inline int starpu_bitmap_cardinal(struct starpu_bitmap *b);

/** return the index of the first set bit of \p b, -1 if none */
static inline int starpu_bitmap_first(struct starpu_bitmap *b);
/** return the position of the last set bit of \p b, -1 if none */
static inline int starpu_bitmap_last(struct starpu_bitmap *b);
/** return the position of set bit right after \p e in \p b, -1 if none */
static inline int starpu_bitmap_next(struct starpu_bitmap *b, int e);
/** todo */
static inline int starpu_bitmap_has_next(struct starpu_bitmap *b, int e);

/** @} */

/**
   todo
*/
struct starpu_bitmap
{
	unsigned long bits[_STARPU_BITMAP_SIZE];
	int cardinal;
};

#ifdef _STARPU_DEBUG_BITMAP
static int _starpu_check_bitmap(struct starpu_bitmap *b)
{
	int card = b->cardinal;
	int i	 = starpu_bitmap_first(b);
	int j;
	for (j = 0; j < card; j++)
	{
		if (i == -1)
			return 0;
		int tmp = starpu_bitmap_next(b, i);
		if (tmp == i)
			return 0;
		i = tmp;
	}
	if (i != -1)
		return 0;
	return 1;
}
#else
#define _starpu_check_bitmap(b) 1
#endif

static int _starpu_count_bit_static(unsigned long e)
{
#if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__) >= 4)
	return __builtin_popcountl(e);
#else
	int c = 0;
	while (e)
	{
		c += e & 1;
		e >>= 1;
	}
	return c;
#endif
}

static inline struct starpu_bitmap *starpu_bitmap_create(void)
{
	return (struct starpu_bitmap *)calloc(1, sizeof(struct starpu_bitmap));
}

static inline void starpu_bitmap_init(struct starpu_bitmap *b)
{
	memset(b, 0, sizeof(*b));
}

static inline void starpu_bitmap_destroy(struct starpu_bitmap *b)
{
	free(b);
}

static inline void starpu_bitmap_set(struct starpu_bitmap *b, int e)
{
	if (!starpu_bitmap_get(b, e))
		b->cardinal++;
	else
		return;
	STARPU_ASSERT(e / _STARPU_LONG_BIT < _STARPU_BITMAP_SIZE);
	b->bits[e / _STARPU_LONG_BIT] |= (1ul << (e % _STARPU_LONG_BIT));
	STARPU_ASSERT(_starpu_check_bitmap(b));
}
static inline void starpu_bitmap_unset(struct starpu_bitmap *b, int e)
{
	if (starpu_bitmap_get(b, e))
		b->cardinal--;
	else
		return;
	STARPU_ASSERT(e / _STARPU_LONG_BIT < _STARPU_BITMAP_SIZE);
	if (e / _STARPU_LONG_BIT > _STARPU_BITMAP_SIZE)
		return;
	b->bits[e / _STARPU_LONG_BIT] &= ~(1ul << (e % _STARPU_LONG_BIT));
	STARPU_ASSERT(_starpu_check_bitmap(b));
}

static inline void starpu_bitmap_unset_all(struct starpu_bitmap *b)
{
	memset(b->bits, 0, _STARPU_BITMAP_SIZE * sizeof(unsigned long));
}

static inline void starpu_bitmap_unset_and(struct starpu_bitmap *a, struct starpu_bitmap *b, struct starpu_bitmap *c)
{
	a->cardinal = 0;
	int i;
	for (i = 0; i < _STARPU_BITMAP_SIZE; i++)
	{
		a->bits[i] = b->bits[i] & c->bits[i];
		a->cardinal += _starpu_count_bit_static(a->bits[i]);
	}
}

static inline int starpu_bitmap_get(struct starpu_bitmap *b, int e)
{
	STARPU_ASSERT(e / _STARPU_LONG_BIT < _STARPU_BITMAP_SIZE);
	if (e / _STARPU_LONG_BIT >= _STARPU_BITMAP_SIZE)
		return 0;
	return (b->bits[e / _STARPU_LONG_BIT] & (1ul << (e % _STARPU_LONG_BIT))) ? 1 : 0;
}

static inline void starpu_bitmap_or(struct starpu_bitmap *a, struct starpu_bitmap *b)
{
	int i;
	a->cardinal = 0;
	for (i = 0; i < _STARPU_BITMAP_SIZE; i++)
	{
		a->bits[i] |= b->bits[i];
		a->cardinal += _starpu_count_bit_static(a->bits[i]);
	}
}

static inline int starpu_bitmap_and_get(struct starpu_bitmap *b1, struct starpu_bitmap *b2, int e)
{
	return starpu_bitmap_get(b1, e) && starpu_bitmap_get(b2, e);
}

static inline int starpu_bitmap_cardinal(struct starpu_bitmap *b)
{
	return b->cardinal;
}

static inline int _starpu_get_first_bit_rank(unsigned long ms)
{
	STARPU_ASSERT(ms != 0);
#if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4))
	return __builtin_ffsl(ms) - 1;
#else
	unsigned long m = 1ul;
	int i		= 0;
	while (!(m & ms))
		i++, m <<= 1;
	return i;
#endif
}

static inline int _starpu_get_last_bit_rank(unsigned long l)
{
	STARPU_ASSERT(l != 0);
#if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4))
	return 8 * sizeof(l) - __builtin_clzl(l);
#else
	int ibit = _STARPU_LONG_BIT - 1;
	while ((!(1ul << ibit)) & l)
		ibit--;
	STARPU_ASSERT(ibit >= 0);
	return ibit;
#endif
}

static inline int starpu_bitmap_first(struct starpu_bitmap *b)
{
	int i = 0;
	while (i < _STARPU_BITMAP_SIZE && !b->bits[i])
		i++;
	if (i == _STARPU_BITMAP_SIZE)
		return -1;
	int nb_long	 = i;
	unsigned long ms = b->bits[i];

	return (nb_long * _STARPU_LONG_BIT) + _starpu_get_first_bit_rank(ms);
}

static inline int starpu_bitmap_has_next(struct starpu_bitmap *b, int e)
{
	int nb_long	   = (e + 1) / _STARPU_LONG_BIT;
	int nb_bit	   = (e + 1) % _STARPU_LONG_BIT;
	unsigned long mask = (~0ul) << nb_bit;
	if (b->bits[nb_long] & mask)
		return 1;
	for (nb_long++; nb_long < _STARPU_BITMAP_SIZE; nb_long++)
		if (b->bits[nb_long])
			return 1;
	return 0;
}

static inline int starpu_bitmap_last(struct starpu_bitmap *b)
{
	if (b->cardinal == 0)
		return -1;
	int ilong;
	for (ilong = _STARPU_BITMAP_SIZE - 1; ilong >= 0; ilong--)
	{
		if (b->bits[ilong])
			break;
	}
	STARPU_ASSERT(ilong >= 0);
	unsigned long l = b->bits[ilong];
	return ilong * _STARPU_LONG_BIT + _starpu_get_last_bit_rank(l);
}

static inline int starpu_bitmap_next(struct starpu_bitmap *b, int e)
{
	int nb_long	   = e / _STARPU_LONG_BIT;
	int nb_bit	   = e % _STARPU_LONG_BIT;
	unsigned long rest = nb_bit == _STARPU_LONG_BIT - 1 ? 0 : (~0ul << (nb_bit + 1)) & b->bits[nb_long];
	if (nb_bit != (_STARPU_LONG_BIT - 1) && rest)
	{
		int i = _starpu_get_first_bit_rank(rest);
		STARPU_ASSERT(i >= 0 && i < _STARPU_LONG_BIT);
		return (nb_long * _STARPU_LONG_BIT) + i;
	}

	for (nb_long++; nb_long < _STARPU_BITMAP_SIZE; nb_long++)
		if (b->bits[nb_long])
			return nb_long * _STARPU_LONG_BIT + _starpu_get_first_bit_rank(b->bits[nb_long]);
	return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_BITMAP_H__ */
