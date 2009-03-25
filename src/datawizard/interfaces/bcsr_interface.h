#ifndef __BCSR_INTERFACE_H__
#define __BCSR_INTERFACE_H__

#include <stdint.h>

/* this interface is used for Sparse matrices */

#define BCSR_INTERFACE	0x118504

typedef struct bcsr_interface_s {
	uint32_t nnz; /* number of non-zero BLOCKS */
	uint32_t nrow; /* number of rows (in terms of BLOCKS) */

	uintptr_t nzval; /* non-zero values */
	uint32_t *colind; /* position of non-zero entried on the row */
//	uint32_t *rowind; /* position of non-zero entried on the col */
	uint32_t *rowptr; /* index (in nzval) of the first entry of the row */

        /* k for k-based indexing (0 or 1 usually) */
        /* also useful when partitionning the matrix ... */
        uint32_t firstentry;

	/* size of the blocks */
	uint32_t r;
	uint32_t c;

	size_t elemsize;
} bcsr_interface_t;

struct data_state_t;

void monitor_bcsr_data(struct data_state_t **handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, uint32_t r, uint32_t c, size_t elemsize);


uint32_t get_bcsr_nnz(struct data_state_t *state);
uint32_t get_bcsr_nrow(struct data_state_t *state);
uint32_t get_bcsr_firstentry(struct data_state_t *state);
uintptr_t get_bcsr_local_nzval(struct data_state_t *state);
uint32_t *get_bcsr_local_colind(struct data_state_t *state);
uint32_t *get_bcsr_local_rowptr(struct data_state_t *state);
uint32_t get_bcsr_r(struct data_state_t *state);
uint32_t get_bcsr_c(struct data_state_t *state);

#endif // __BCSR_INTERFACE_H__
