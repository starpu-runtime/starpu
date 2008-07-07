#ifndef __CRS_INTERFACE_H__
#define __CRS_INTERFACE_H__

/* this interface is used for Sparse matrices */

#define CRS_INTERFACE	0x118502

typedef struct crs_interface_s {
	int nnz; /* number of non-zero entries */
	int nrow; /* number of rows */
	float *nzval; /* non-zero values */
	int *colind; /* position of non-zero entried on the row */
	int *rowptr; /* index (in nzval) of the first entry of the row */
	int firstentry; /* XXX useful when partitionning the matrix ... */
} crs_interface_t;

#endif // __CRS_INTERFACE_H__
