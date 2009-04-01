/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __STARPU_DATA_INTERFACES_H__
#define __STARPU_DATA_INTERFACES_H__

struct starpu_data_state_t;
typedef struct starpu_data_state_t * starpu_data_handle;

/* BLAS interface for dense matrices */
typedef struct starpu_blas_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t ny;
	uint32_t ld;
	size_t elemsize;
} starpu_blas_interface_t;

void starpu_monitor_blas_data(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t ld, uint32_t nx,
                        uint32_t ny, size_t elemsize);
uint32_t starpu_get_blas_nx(starpu_data_handle handle);
uint32_t starpu_get_blas_ny(starpu_data_handle handle);
uint32_t starpu_get_blas_local_ld(starpu_data_handle handle);
uintptr_t starpu_get_blas_local_ptr(starpu_data_handle handle);

/* vector interface for contiguous (non-strided) buffers */
typedef struct starpu_vector_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	size_t elemsize;
} starpu_vector_interface_t;

void starpu_monitor_vector_data(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize);
uint32_t starpu_get_vector_nx(starpu_data_handle handle);
uintptr_t starpu_get_vector_local_ptr(starpu_data_handle handle);

/* CSR interface for sparse matrices (compressed sparse row representation) */
typedef struct starpu_csr_interface_s {
	uint32_t nnz; /* number of non-zero entries */
	uint32_t nrow; /* number of rows */
	uintptr_t nzval; /* non-zero values */
	uint32_t *colind; /* position of non-zero entried on the row */
	uint32_t *rowptr; /* index (in nzval) of the first entry of the row */

        /* k for k-based indexing (0 or 1 usually) */
        /* also useful when partitionning the matrix ... */
        uint32_t firstentry;

	size_t elemsize;
} starpu_csr_interface_t;

void starpu_monitor_csr_data(starpu_data_handle *handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize);
uint32_t starpu_get_csr_nnz(starpu_data_handle handle);
uint32_t starpu_get_csr_nrow(starpu_data_handle handle);
uint32_t starpu_get_csr_firstentry(starpu_data_handle handle);
uintptr_t starpu_get_csr_local_nzval(starpu_data_handle handle);
uint32_t *starpu_get_csr_local_colind(starpu_data_handle handle);
uint32_t *starpu_get_csr_local_rowptr(starpu_data_handle handle);

/* CSC interface for sparse matrices (compressed sparse column representation) */
typedef struct starpu_csc_interface_s {
	int nnz; /* number of non-zero entries */
	int nrow; /* number of rows */
	float *nzval; /* non-zero values */
	int *colind; /* position of non-zero entried on the row */
	int *rowptr; /* index (in nzval) of the first entry of the row */

	/* k for k-based indexing (0 or 1 usually) */
	/* also useful when partitionning the matrix ... */
	int firstentry; 
} starpu_csc_interface_t;

/* BCSR interface for sparse matrices (blocked compressed sparse row
 * representation) */
typedef struct starpu_bcsr_interface_s {
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
} starpu_bcsr_interface_t;

void starpu_monitor_bcsr_data(starpu_data_handle *handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, uint32_t r, uint32_t c, size_t elemsize);


uint32_t starpu_get_bcsr_nnz(starpu_data_handle);
uint32_t starpu_get_bcsr_nrow(starpu_data_handle);
uint32_t starpu_get_bcsr_firstentry(starpu_data_handle);
uintptr_t starpu_get_bcsr_local_nzval(starpu_data_handle);
uint32_t *starpu_get_bcsr_local_colind(starpu_data_handle);
uint32_t *starpu_get_bcsr_local_rowptr(starpu_data_handle);
uint32_t starpu_get_bcsr_r(starpu_data_handle);
uint32_t starpu_get_bcsr_c(starpu_data_handle);

typedef union {
	starpu_blas_interface_t blas;	/* dense BLAS representation */
	starpu_vector_interface_t vector; /* continuous vector */
	starpu_csr_interface_t csr;	/* compressed sparse row */
	starpu_csc_interface_t csc; 	/* compressed sparse column */
	starpu_bcsr_interface_t bcsr;	/* blocked compressed sparse row */
} starpu_data_interface_t;


#endif // __STARPU_DATA_INTERFACES_H__
