#ifndef __BITMAP_H__
#define __BITMAP_H__

struct _starpu_bitmap;

struct _starpu_bitmap * _starpu_bitmap_create(void);
void _starpu_bitmap_destroy(struct _starpu_bitmap *);

void _starpu_bitmap_set(struct _starpu_bitmap *, int);
void _starpu_bitmap_unset(struct _starpu_bitmap *, int);
void _starpu_bitmap_unset_all(struct _starpu_bitmap *);

int _starpu_bitmap_get(struct _starpu_bitmap *, int);

//this is basically compute a |= b;
void _starpu_bitmap_or(struct _starpu_bitmap * a,
		       struct _starpu_bitmap * b);

//return 1 iff e set in b1 AND e set in b2
int _starpu_bitmap_and_get(struct _starpu_bitmap * b1,
			   struct _starpu_bitmap * b2,
			   int e);

int _starpu_bitmap_cardinal(struct _starpu_bitmap *);

//return the index of first bit, -1 if none
int _starpu_bitmap_first(struct _starpu_bitmap *);
int _starpu_bitmap_last(struct _starpu_bitmap *);
//return the index of bit right after e, -1 if none
int _starpu_bitmap_next(struct _starpu_bitmap *, int e);

#endif
