#ifndef __CSR_INTERFACE_H__
#define __CSR_INTERFACE_H__

/* this interface is used for Sparse matrices */

#define CSR_INTERFACE	0x118502

typedef struct csr_interface_s {
	int nnz; /* number of non-zero entries */
	int nrow; /* number of rows */
	float *nzval; /* non-zero values */
	int *colind; /* position of non-zero entried on the row */
	int *rowptr; /* index (in nzval) of the first entry of the row */

        /* k for k-based indexing (0 or 1 usually) */
        /* also useful when partitionning the matrix ... */
        int firstentry;
} csr_interface_t;

#endif // __CSR_INTERFACE_H__
