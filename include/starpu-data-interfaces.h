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

#include <starpu.h>
#include <starpu-data.h>

#ifdef __cplusplus
extern "C" {
#endif

void *starpu_data_get_interface_on_node(starpu_data_handle handle, unsigned memory_node);

/* BLAS interface for dense matrices */
typedef struct starpu_blas_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t ny;
	uint32_t ld;
	size_t elemsize;
} starpu_blas_interface_t;

void starpu_register_blas_data(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t ld, uint32_t nx,
                        uint32_t ny, size_t elemsize);
uint32_t starpu_get_blas_nx(starpu_data_handle handle);
uint32_t starpu_get_blas_ny(starpu_data_handle handle);
uint32_t starpu_get_blas_local_ld(starpu_data_handle handle);
uintptr_t starpu_get_blas_local_ptr(starpu_data_handle handle);
size_t starpu_get_blas_elemsize(starpu_data_handle handle);

/* helper methods */
#define GET_BLAS_PTR(interface)	(((starpu_blas_interface_t *)(interface))->ptr)
#define GET_BLAS_NX(interface)	(((starpu_blas_interface_t *)(interface))->nx)
#define GET_BLAS_NY(interface)	(((starpu_blas_interface_t *)(interface))->ny)
#define GET_BLAS_LD(interface)	(((starpu_blas_interface_t *)(interface))->ld)
#define GET_BLAS_ELEMSIZE(interface)	(((starpu_blas_interface_t *)(interface))->elemsize)


/* BLOCK interface for 3D dense blocks */
typedef struct starpu_block_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	uint32_t ny;
	uint32_t nz;
	uint32_t ldy;	/* number of elements between two lines */
	uint32_t ldz;	/* number of elements between two planes */
	size_t elemsize;
} starpu_block_interface_t;

void starpu_register_block_data(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t nx,
                        uint32_t ny, uint32_t nz, size_t elemsize);
uint32_t starpu_get_block_nx(starpu_data_handle handle);
uint32_t starpu_get_block_ny(starpu_data_handle handle);
uint32_t starpu_get_block_nz(starpu_data_handle handle);
uint32_t starpu_get_block_local_ldy(starpu_data_handle handle);
uint32_t starpu_get_block_local_ldz(starpu_data_handle handle);
uintptr_t starpu_get_block_local_ptr(starpu_data_handle handle);
size_t starpu_get_block_elemsize(starpu_data_handle handle);

/* helper methods */
#define STARPU_GET_BLOCK_PTR(interface)	(((starpu_block_interface_t *)(interface))->ptr)
#define STARPU_GET_BLOCK_NX(interface)	(((starpu_block_interface_t *)(interface))->nx)
#define STARPU_GET_BLOCK_NY(interface)	(((starpu_block_interface_t *)(interface))->ny)
#define STARPU_GET_BLOCK_NZ(interface)	(((starpu_block_interface_t *)(interface))->nz)
#define STARPU_GET_BLOCK_LDY(interface)	(((starpu_block_interface_t *)(interface))->ldy)
#define STARPU_GET_BLOCK_LDZ(interface)	(((starpu_block_interface_t *)(interface))->ldz)
#define STARPU_GET_BLOCK_ELEMSIZE(interface)	(((starpu_block_interface_t *)(interface))->elemsize)

/* vector interface for contiguous (non-strided) buffers */
typedef struct starpu_vector_interface_s {
	uintptr_t ptr;
	uint32_t nx;
	size_t elemsize;
} starpu_vector_interface_t;

void starpu_register_vector_data(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize);
uint32_t starpu_get_vector_nx(starpu_data_handle handle);
size_t starpu_get_vector_elemsize(starpu_data_handle handle);
uintptr_t starpu_get_vector_local_ptr(starpu_data_handle handle);

/* helper methods */
#define STARPU_GET_VECTOR_PTR(interface)	(((starpu_vector_interface_t *)(interface))->ptr)
#define STARPU_GET_VECTOR_NX(interface)	(((starpu_vector_interface_t *)(interface))->nx)
#define STARPU_GET_VECTOR_ELEMSIZE(interface)	(((starpu_vector_interface_t *)(interface))->elemsize)

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

void starpu_register_csr_data(starpu_data_handle *handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize);
uint32_t starpu_get_csr_nnz(starpu_data_handle handle);
uint32_t starpu_get_csr_nrow(starpu_data_handle handle);
uint32_t starpu_get_csr_firstentry(starpu_data_handle handle);
uintptr_t starpu_get_csr_local_nzval(starpu_data_handle handle);
uint32_t *starpu_get_csr_local_colind(starpu_data_handle handle);
uint32_t *starpu_get_csr_local_rowptr(starpu_data_handle handle);
size_t starpu_get_csr_elemsize(starpu_data_handle handle);

#define STARPU_GET_CSR_NNZ(interface)	(((starpu_csr_interface_t *)(interface))->nnz)
#define STARPU_GET_CSR_NROW(interface)	(((starpu_csr_interface_t *)(interface))->nrow)
#define STARPU_GET_CSR_NZVAL(interface)	(((starpu_csr_interface_t *)(interface))->nzval)
#define STARPU_GET_CSR_COLIND(interface)	(((starpu_csr_interface_t *)(interface))->colind)
#define STARPU_GET_CSR_ROWPTR(interface)	(((starpu_csr_interface_t *)(interface))->rowptr)
#define STARPU_GET_CSR_FIRSTENTRY(interface)	(((starpu_csr_interface_t *)(interface))->firstentry)
#define STARPU_GET_CSR_ELEMSIZE(interface)	(((starpu_csr_interface_t *)(interface))->elemsize)

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

void starpu_register_bcsr_data(starpu_data_handle *handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, uint32_t r, uint32_t c, size_t elemsize);


uint32_t starpu_get_bcsr_nnz(starpu_data_handle);
uint32_t starpu_get_bcsr_nrow(starpu_data_handle);
uint32_t starpu_get_bcsr_firstentry(starpu_data_handle);
uintptr_t starpu_get_bcsr_local_nzval(starpu_data_handle);
uint32_t *starpu_get_bcsr_local_colind(starpu_data_handle);
uint32_t *starpu_get_bcsr_local_rowptr(starpu_data_handle);
uint32_t starpu_get_bcsr_r(starpu_data_handle);
uint32_t starpu_get_bcsr_c(starpu_data_handle);
size_t starpu_get_bcsr_elemsize(starpu_data_handle);

#define STARPU_BLAS_INTERFACE_ID	0
#define STARPU_BLOCK_INTERFACE_ID	1
#define STARPU_VECTOR_INTERFACE_ID	2
#define STARPU_CSR_INTERFACE_ID		3
#define STARPU_CSC_INTERFACE_ID		4
#define STARPU_BCSCR_INTERFACE_ID	5
#define STARPU_NINTERFACES_ID		6 /* number of data interfaces */

unsigned starpu_get_handle_interface_id(starpu_data_handle);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_DATA_INTERFACES_H__
