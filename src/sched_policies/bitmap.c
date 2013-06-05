#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <starpu.h>
#include "bitmap.h"

#ifndef LONG_BIT
#define LONG_BIT (sizeof(unsigned long) * 8)
#endif



struct _starpu_bitmap{
	unsigned long * bits;
	int size;
	int cardinal;
};

#ifndef STARPU_NO_ASSERT
static int check_bitmap(struct _starpu_bitmap *b)
{
	int card = b->cardinal;
	int i = _starpu_bitmap_first(b);
	int j;
	for(j = 0; j < card; j++)
	{
		if(i == -1)
			return 0;
		int tmp = _starpu_bitmap_next(b,i);
		if(tmp == i)
			return 0;
		i = tmp;
	}
	if(i != -1)
		return 0;
	return 1;
}
#endif


struct _starpu_bitmap * _starpu_bitmap_create(void)
{
	struct _starpu_bitmap * b = malloc(sizeof(*b));
	memset(b,0,sizeof(*b));
	return b;
}
void _starpu_bitmap_destroy(struct _starpu_bitmap * b)
{

	free(b->bits);
	free(b);
}

void _starpu_bitmap_set(struct _starpu_bitmap * b, int e)
{
	if(!_starpu_bitmap_get(b, e))
		b->cardinal++;
	if((e/LONG_BIT) + 1 > b->size)
	{
		b->bits = realloc(b->bits, sizeof(unsigned long) * ((e/LONG_BIT) + 1));
		memset(b->bits + b->size, 0, sizeof(unsigned long) * ((e/LONG_BIT + 1) - b->size));
		b->size = (e/LONG_BIT) + 1;
	}
	b->bits[e/LONG_BIT] |= (1ul << (e%LONG_BIT));
	STARPU_ASSERT(check_bitmap(b));

}
void _starpu_bitmap_unset(struct _starpu_bitmap *b, int e)
{
	if(_starpu_bitmap_get(b, e))
		b->cardinal--;
	else
		return;
	if(e / LONG_BIT > b->size)
		return;
	b->bits[e/LONG_BIT] &= ~(1ul << (e%LONG_BIT));
	STARPU_ASSERT(check_bitmap(b));

}

void _starpu_bitmap_unset_all(struct _starpu_bitmap * b)
{
	free(b->bits);
	b->bits = NULL;
	b->size = 0;
	STARPU_ASSERT(check_bitmap(b));

}

int _starpu_bitmap_get(struct _starpu_bitmap * b, int e)
{
	if(e / LONG_BIT >= b->size)
		return 0;
	return b->bits[e/LONG_BIT] & (1 << (e%LONG_BIT));
	STARPU_ASSERT(check_bitmap(b));
}

void _starpu_bitmap_or(struct _starpu_bitmap * a, struct _starpu_bitmap * b)
{
	if(a->size < b->size)
	{
		a->bits = realloc(a->bits, b->size * sizeof(unsigned long));
		a->size = b->size;
	}
	int i;
	for(i = 0; i < b->size; i++)
	{
		a->bits[i] |= b->bits[i];
	}
	STARPU_ASSERT(check_bitmap(b));
}


int _starpu_bitmap_and_get(struct _starpu_bitmap * b1, struct _starpu_bitmap * b2, int e)
{
	return _starpu_bitmap_get(b1,e) && _starpu_bitmap_get(b2,e);
}

int _starpu_bitmap_cardinal(struct _starpu_bitmap * b)
{
	return b->cardinal;
}


static inline int get_first_bit_rank(unsigned long ms)
{
	STARPU_ASSERT(ms != 0);
	unsigned long m = 1ul;
	int i = 0;
	while(!(m&ms))
		i++,m<<=1;
	return i;
}


int _starpu_bitmap_first(struct _starpu_bitmap * b)
{
	int i = 0;
	while(i < b->size && !b->bits[i])
		i++;
	if( i == b->size)
		return -1;
	int nb_long = i;
	unsigned long ms = b->bits[i];
	int m = 1;
	i = 0;
	while(1)
		if(m&ms)
			return (nb_long * LONG_BIT) + i;
		else
		{
			i++;
			m<<=1;
		}
	STARPU_ASSERT_MSG(0, "this should never be reached");
}

int _starpu_bitmap_has_next(struct _starpu_bitmap * b, int e)
{
	int nb_long = e / LONG_BIT;
	int nb_bit = e % LONG_BIT;
	unsigned long mask = (~0ul) << (nb_bit + 1);
	if(b->bits[nb_long] & mask)
		return 1;
	for(nb_long++; nb_long < b->size; nb_long++)
		if(b->bits[nb_long])
			return 1;
	return 0;
}

int _starpu_bitmap_last(struct _starpu_bitmap * b)
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
	int ibit = LONG_BIT - 1;
	while((!(1ul << ibit)) & l)
		ibit--;
	STARPU_ASSERT(ibit >= 0);
	return ilong * LONG_BIT + ibit;
}

int _starpu_bitmap_next(struct _starpu_bitmap *b, int e)
{
	int nb_long = e / LONG_BIT;
	int nb_bit = e % LONG_BIT;
	if((~0ul << (nb_bit + 1)) & b->bits[nb_long])
	{
		unsigned long mask = 1ul<<nb_bit;
		int i = nb_bit + 1;
		while(1)
		{
			if(b->bits[nb_long] & (mask<<i))
				return (nb_long * LONG_BIT) + nb_bit + i;
			i++;
		}
	}

	for(nb_long++;nb_long < b->size; nb_long++)
		if(b->bits[nb_long])
			return nb_long * LONG_BIT + get_first_bit_rank(b->bits[nb_long]);
	return -1;
}
