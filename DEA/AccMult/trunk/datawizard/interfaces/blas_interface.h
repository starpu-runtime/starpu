#ifndef __BLAS_INTERFACE_H__
#define __BLAS_INTERFACE_H__

#include <stdint.h>

#define BLAS_INTERFACE   0x118501

typedef struct blas_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t ny;
	uint32_t ld;
	size_t elemsize;
} blas_interface_t;

#endif // __BLAS_INTERFACE_H__
