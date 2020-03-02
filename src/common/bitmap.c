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

#include <starpu.h>
#include <starpu_bitmap.h>

#include <limits.h>
#include <string.h>
#include <stdlib.h>

#ifndef LONG_BIT
#define LONG_BIT (sizeof(unsigned long) * 8)
#endif

struct starpu_bitmap
{
	unsigned long * bits;
	int size; /* the size of bits array in number of unsigned long */
	int cardinal;
};

//#define DEBUG_BITMAP

#ifdef DEBUG_BITMAP
static int check_bitmap(struct starpu_bitmap *b)
{
	int card = b->cardinal;
	int i = starpu_bitmap_first(b);
	int j;
	for(j = 0; j < card; j++)
	{
		if(i == -1)
			return 0;
		int tmp = starpu_bitmap_next(b,i);
		if(tmp == i)
			return 0;
		i = tmp;
	}
	if(i != -1)
		return 0;
	return 1;
}
#else
#define check_bitmap(b) 1
#endif

static int _count_bit(unsigned long e)
{
#if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__) >= 4)
	return __builtin_popcountl(e);
#else
	int c = 0;
	while(e)
	{
		c += e&1;
		e >>= 1;
	}
	return c;
#endif
}

struct starpu_bitmap * starpu_bitmap_create(void)
{
	struct starpu_bitmap *b;
	_STARPU_CALLOC(b, 1, sizeof(*b));
	return b;
}
void starpu_bitmap_destroy(struct starpu_bitmap * b)
{
	if(b)
	{
		free(b->bits);
		free(b);
	}
}

void starpu_bitmap_set(struct starpu_bitmap * b, int e)
{

	if(!starpu_bitmap_get(b, e))
		b->cardinal++;
	else
		return;
	if((e/LONG_BIT) + 1 > b->size)
	{
		_STARPU_REALLOC(b->bits, sizeof(unsigned long) * ((e/LONG_BIT) + 1));
		memset(b->bits + b->size, 0, sizeof(unsigned long) * ((e/LONG_BIT + 1) - b->size));
		b->size = (e/LONG_BIT) + 1;
	}
	b->bits[e/LONG_BIT] |= (1ul << (e%LONG_BIT));
	STARPU_ASSERT(check_bitmap(b));
}
void starpu_bitmap_unset(struct starpu_bitmap *b, int e)
{
	if(starpu_bitmap_get(b, e))
		b->cardinal--;
	else
		return;
	if(e / LONG_BIT > b->size)
		return;
	b->bits[e/LONG_BIT] &= ~(1ul << (e%LONG_BIT));
	STARPU_ASSERT(check_bitmap(b));
}

void starpu_bitmap_unset_all(struct starpu_bitmap * b)
{
	free(b->bits);
	b->bits = NULL;
	b->size = 0;
}

void starpu_bitmap_unset_and(struct starpu_bitmap * a, struct starpu_bitmap * b, struct starpu_bitmap * c)
{
	int n = STARPU_MIN(b->size, c->size);
	_STARPU_REALLOC(a->bits, sizeof(unsigned long) * n);
	a->size = n;
	a->cardinal = 0;
	int i;
	for(i = 0; i < n; i++)
	{
		a->bits[i] = b->bits[i] & c->bits[i];
		a->cardinal += _count_bit(a->bits[i]);
	}
}

int starpu_bitmap_get(struct starpu_bitmap * b, int e)
{
	if(e / LONG_BIT >= b->size)
		return 0;
	return (b->bits[e/LONG_BIT] & (1ul << (e%LONG_BIT))) ?
		1:
		0;
}

void starpu_bitmap_or(struct starpu_bitmap * a, struct starpu_bitmap * b)
{
	if(a->size < b->size)
	{
		_STARPU_REALLOC(a->bits, b->size * sizeof(unsigned long));
		memset(a->bits + a->size, 0, (b->size - a->size) * sizeof(unsigned long));
		a->size = b->size;

	}
	int i;
	for(i = 0; i < b->size; i++)
	{
		a->bits[i] |= b->bits[i];
	}
	a->cardinal = 0;
	for(i = 0; i < a->size; i++)
		a->cardinal += _count_bit(a->bits[i]);
}


int starpu_bitmap_and_get(struct starpu_bitmap * b1, struct starpu_bitmap * b2, int e)
{
	return starpu_bitmap_get(b1,e) && starpu_bitmap_get(b2,e);
}

int starpu_bitmap_cardinal(struct starpu_bitmap * b)
{
	return b->cardinal;
}


static inline int get_first_bit_rank(unsigned long ms)
{
	STARPU_ASSERT(ms != 0);
#if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4))
	return __builtin_ffsl(ms) - 1;
#else
	unsigned long m = 1ul;
	int i = 0;
	while(!(m&ms))
		i++,m<<=1;
	return i;
#endif
}

static inline int get_last_bit_rank(unsigned long l)
{
	STARPU_ASSERT(l != 0);
#if (__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4))
	return 8*sizeof(l) - __builtin_clzl(l);
#else
	int ibit = LONG_BIT - 1;
	while((!(1ul << ibit)) & l)
		ibit--;
	STARPU_ASSERT(ibit >= 0);
	return ibit;
#endif
}

int starpu_bitmap_first(struct starpu_bitmap * b)
{
	int i = 0;
	while(i < b->size && !b->bits[i])
		i++;
	if( i == b->size)
		return -1;
	int nb_long = i;
	unsigned long ms = b->bits[i];

	return (nb_long * LONG_BIT) + get_first_bit_rank(ms);
}

int starpu_bitmap_has_next(struct starpu_bitmap * b, int e)
{
	int nb_long = (e+1) / LONG_BIT;
	int nb_bit = (e+1) % LONG_BIT;
	unsigned long mask = (~0ul) << nb_bit;
	if(b->bits[nb_long] & mask)
		return 1;
	for(nb_long++; nb_long < b->size; nb_long++)
		if(b->bits[nb_long])
			return 1;
	return 0;
}

int starpu_bitmap_last(struct starpu_bitmap * b)
{
	if(b->cardinal == 0)
		return -1;
	int ilong;
	for(ilong = b->size - 1; ilong >= 0; ilong--)
	{
		if(b->bits[ilong])
			break;
	}
	STARPU_ASSERT(ilong >= 0);
	unsigned long l = b->bits[ilong];
	return ilong * LONG_BIT + get_last_bit_rank(l);
}

int starpu_bitmap_next(struct starpu_bitmap *b, int e)
{
	int nb_long = e / LONG_BIT;
	int nb_bit = e % LONG_BIT;
	unsigned long rest = nb_bit == LONG_BIT - 1 ? 0 : (~0ul << (nb_bit + 1)) & b->bits[nb_long];
	if(nb_bit != (LONG_BIT - 1) && rest)
	{
		int i = get_first_bit_rank(rest);
		STARPU_ASSERT(i >= 0 && i < LONG_BIT);
		return (nb_long * LONG_BIT) + i;
	}

	for(nb_long++;nb_long < b->size; nb_long++)
		if(b->bits[nb_long])
			return nb_long * LONG_BIT + get_first_bit_rank(b->bits[nb_long]);
	return -1;
}
