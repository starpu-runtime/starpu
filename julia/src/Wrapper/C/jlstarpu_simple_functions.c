#include "jlstarpu.h"




int jlstarpu_init(void)
{
	return starpu_init(NULL);
}



void jlstarpu_set_to_zero(void * ptr, unsigned int size)
{
	memset(ptr, 0, size);
}
