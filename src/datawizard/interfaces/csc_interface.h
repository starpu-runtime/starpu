#ifndef __CSC_INTERFACE_H__
#define __CSC_INTERFACE_H__

/* this interface is used for Sparse matrices */

#define CSC_INTERFACE	0x118503

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

#endif // __CSC_INTERFACE_H__
