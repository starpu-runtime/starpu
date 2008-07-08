#include "incrementer_cuda.h"

extern "C" __global__ 
void cuda_incrementer(uint32_t ptr, uint32_t nx, uint32_t ny, uint32_t ld, 
			uint32_t ptr2, uint32_t nx2, uint32_t ny2, uint32_t ld2)
{
	float *tab;
	float *unity;

	tab = (float *)ptr;
	unity = (float *)ptr2;

	tab[0] = tab[0] + unity[0];
	tab[1] = tab[1] + unity[1];
	tab[2] = tab[2] + unity[2];

	return;
}
