
#include <limits.h>

struct _starpu_bitmap{
	unsigned long * bits;
	int size;
};

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


struct tuple {
	int nb_longs;
	int nb_bits;
};

static inline struct tuple get_tuple(int e)
{
	int i;
	for(i = 0; e > LONG_BIT; e -= LONG_BIT, i++)
		;
	struct tuple t = {i, e};
	return t;
}

void _starpu_bitmap_set(struct _starpu_bitmap * b, int e)
{
	struct tuple t = get_tuple(e);
	if(t.nb_long + 1 > b->size)
	{
		b->bits = realloc(b->bits, sizeof(unsigned long) * (t.nb_long + 1));
		b->size = t.nb_long + 1;
	}
	b->bits[t.nb_long] |= (1ul << t.bn_bits);
}
void _starpu_bitmap_unset(struct _starpu_bitmap *b, int e)
{
	struct tuple t = get_tuple(e);
	if(e / LONG_BIT > b->size)
		return;
	b->bits[t.nb_long] ^= ~(1ul << t.bn_bits);
}

int _starpu_bitmap_get(struct _starpu_bitmap * b, int e)
{
	if(e / LONG_BIT > b->size)
		return 0;
	int i;
	struct tuple t = get_tuple(e);
	return b->bits[t.nb_longs] & (1 << t.nb_bits);
}


int _starpu_bitmap_and_get(struct _starpu_bitmap * b1, struct _starpu_bitmap * b2, int e)
{
	return _starpu_bitmap_get(b1,e) && _starpu_bitmap_get(b2,e);
}


