#ifndef __VECTOR_INTERFACE_H__
#define __VECTOR_INTERFACE_H__

#include <stdint.h>

#define VECTOR_INTERFACE   0x118503

typedef struct vector_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	size_t elemsize;
} vector_interface_t;

struct data_state_t;
void monitor_vector_data(struct data_state_t *state, uint32_t home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize);

uint32_t get_vector_nx(struct data_state_t *state);
uintptr_t get_blas_local_ptr(struct data_state_t *state);

#endif // __VECTOR_INTERFACE_H__
