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

struct data_state_t;
void monitor_blas_data(struct data_state_t *state, uint32_t home_node,
                        uintptr_t ptr, uint32_t ld, uint32_t nx,
                        uint32_t ny, size_t elemsize);

uint32_t get_blas_nx(struct data_state_t *state);
uint32_t get_blas_ny(struct data_state_t *state);
uint32_t get_blas_local_ld(struct data_state_t *state);
uintptr_t get_blas_local_ptr(struct data_state_t *state);

#endif // __BLAS_INTERFACE_H__
