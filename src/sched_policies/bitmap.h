#ifndef __BITMAP_H__
#define __BITMAP_H__

struct _starpu_bitmap;

struct _starpu_bitmap * _starpu_bitmap_create(void);
void _starpu_bitmap_destroy(struct _starpu_bitmap *);

void _starpu_bitmap_set(struct _starpu_bitmap *, int);
void _starpu_bitmap_unset(struct _starpu_bitmap *, int);

int _starpu_bitmap_get(struct _starpu_bitmap *, int);

//return 1 iff e set in b1 AND e set in b2
int _starpu_bitmap_and_get(struct _starpu_bitmap * b1,
			   struct _starpu_bitmap * b2,
			   int e);



#endif
