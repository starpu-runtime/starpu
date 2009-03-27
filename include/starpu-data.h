#ifndef __STARPU_DATA_H__
#define __STARPU_DATA_H__

#include <starpu-data-filters.h>

#define NMAXBUFS        8

struct data_state_t;
typedef struct data_state_t * data_handle;

typedef enum {
	R,
	W,
	RW
} access_mode;

typedef struct buffer_descr_t {
	data_handle state;
	access_mode mode;
} buffer_descr;

/* BLAS interface for dense matrices */
typedef struct blas_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t ny;
	uint32_t ld;
	size_t elemsize;
} blas_interface_t;

void monitor_blas_data(data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t ld, uint32_t nx,
                        uint32_t ny, size_t elemsize);
uint32_t get_blas_nx(data_handle handle);
uint32_t get_blas_ny(data_handle handle);
uint32_t get_blas_local_ld(data_handle handle);
uintptr_t get_blas_local_ptr(data_handle handle);

/* vector interface for contiguous (non-strided) buffers */
typedef struct vector_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	size_t elemsize;
} vector_interface_t;

void monitor_vector_data(data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize);
uint32_t get_vector_nx(data_handle handle);
uintptr_t get_vector_local_ptr(data_handle handle);

/* CSR interface for sparse matrices (compressed sparse row representation) */
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

void monitor_csr_data(data_handle *handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize);
uint32_t get_csr_nnz(data_handle handle);
uint32_t get_csr_nrow(data_handle handle);
uint32_t get_csr_firstentry(data_handle handle);
uintptr_t get_csr_local_nzval(data_handle handle);
uint32_t *get_csr_local_colind(data_handle handle);
uint32_t *get_csr_local_rowptr(data_handle handle);

/* CSC interface for sparse matrices (compressed sparse column representation) */
typedef struct csc_interface_s {
	int nnz; /* number of non-zero entries */
	int nrow; /* number of rows */
	float *nzval; /* non-zero values */
	int *colind; /* position of non-zero entried on the row */
	int *rowptr; /* index (in nzval) of the first entry of the row */

	/* k for k-based indexing (0 or 1 usually) */
	/* also useful when partitionning the matrix ... */
	int firstentry; 
} csc_interface_t;

/* BCSR interface for sparse matrices (blocked compressed sparse row
 * representation) */
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

void monitor_bcsr_data(data_handle *handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, uint32_t r, uint32_t c, size_t elemsize);


uint32_t get_bcsr_nnz(data_handle);
uint32_t get_bcsr_nrow(data_handle);
uint32_t get_bcsr_firstentry(data_handle);
uintptr_t get_bcsr_local_nzval(data_handle);
uint32_t *get_bcsr_local_colind(data_handle);
uint32_t *get_bcsr_local_rowptr(data_handle);
uint32_t get_bcsr_r(data_handle);
uint32_t get_bcsr_c(data_handle);

typedef union {
	blas_interface_t blas;	/* dense BLAS representation */
	vector_interface_t vector; /* continuous vector */
	csr_interface_t csr;	/* compressed sparse row */
	csc_interface_t csc; 	/* compressed sparse column */
	bcsr_interface_t bcsr;	/* blocked compressed sparse row */
} data_interface_t;

void unpartition_data(struct data_state_t *root_data, uint32_t gathering_node);
void delete_data(struct data_state_t *state);

void advise_if_data_is_important(struct data_state_t *state, unsigned is_important);

void sync_data_with_mem(struct data_state_t *state);

#endif // __STARPU_DATA_H__
