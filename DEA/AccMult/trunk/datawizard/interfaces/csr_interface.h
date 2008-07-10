#ifndef __CSR_INTERFACE_H__
#define __CSR_INTERFACE_H__

#include <stdint.h>

/* this interface is used for Sparse matrices */

#define CSR_INTERFACE	0x118502

typedef struct csr_interface_s {
	uint32_t nnz; /* number of non-zero entries */
	uint32_t nrow; /* number of rows */
	uintptr_t nzval; /* non-zero values */
	uint32_t *colind; /* position of non-zero entried on the row */
	uint32_t *rowptr; /* index (in nzval) of the first entry of the row */

        /* k for k-based indexing (0 or 1 usually) */
        /* also useful when partitionning the matrix ... */
        uint32_t firstentry;

	size_t elemsize;
} csr_interface_t;

struct data_state_t;

uint32_t get_csr_nnz(struct data_state_t *state);
uint32_t get_csr_nrow(struct data_state_t *state);
uint32_t get_csr_firstentry(struct data_state_t *state);
uintptr_t get_csr_local_nzval(struct data_state_t *state);
uint32_t *get_csr_local_colind(struct data_state_t *state);
uint32_t *get_csr_local_rowptr(struct data_state_t *state);

#endif // __CSR_INTERFACE_H__
