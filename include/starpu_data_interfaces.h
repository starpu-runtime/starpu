/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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
#include <starpu_data.h>

#ifdef STARPU_USE_GORDON
/* to get the gordon_strideSize_t data structure from gordon */
#include <gordon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The following structures are used to describe data interfaces */

/* This structure contains the different methods to transfer data between the
 * different types of memory nodes */
struct starpu_data_copy_methods {
	/* src type is ram */
	int (*ram_to_ram)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*ram_to_cuda)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*ram_to_opencl)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*ram_to_spu)(starpu_data_handle handle, uint32_t src, uint32_t dst);

	/* src type is cuda */
	int (*cuda_to_ram)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*cuda_to_cuda)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*cuda_to_opencl)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*cuda_to_spu)(starpu_data_handle handle, uint32_t src, uint32_t dst);

	/* src type is spu */
	int (*spu_to_ram)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*spu_to_cuda)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*spu_to_opencl)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*spu_to_spu)(starpu_data_handle handle, uint32_t src, uint32_t dst);

	/* src type is opencl */
	int (*opencl_to_ram)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*opencl_to_cuda)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*opencl_to_opencl)(starpu_data_handle handle, uint32_t src, uint32_t dst);
	int (*opencl_to_spu)(starpu_data_handle handle, uint32_t src, uint32_t dst);

#ifdef STARPU_USE_CUDA
	/* for asynchronous CUDA transfers */
	int (*ram_to_cuda_async)(starpu_data_handle handle, uint32_t src,
					uint32_t dst, cudaStream_t *stream);
	int (*cuda_to_ram_async)(starpu_data_handle handle, uint32_t src,
					uint32_t dst, cudaStream_t *stream);
	int (*cuda_to_cuda_async)(starpu_data_handle handle, uint32_t src,
					uint32_t dst, cudaStream_t *stream);
#endif

#ifdef STARPU_USE_OPENCL
	/* for asynchronous OpenCL transfers */
        int (*ram_to_opencl_async)(starpu_data_handle handle, uint32_t src, uint32_t dst, cl_event *event);
	int (*opencl_to_ram_async)(starpu_data_handle handle, uint32_t src, uint32_t dst, cl_event *event);
	int (*opencl_to_opencl_async)(starpu_data_handle handle, uint32_t src, uint32_t dst, cl_event *event);
#endif
};

struct starpu_data_interface_ops_t {
	void (*register_data_handle)(starpu_data_handle handle,
					uint32_t home_node, void *interface);
	size_t (*allocate_data_on_node)(void *interface, uint32_t node);
	void (*free_data_on_node)(void *interface, uint32_t node);
	const struct starpu_data_copy_methods *copy_methods;
	size_t (*get_size)(starpu_data_handle handle);
	uint32_t (*footprint)(starpu_data_handle handle);
	int (*compare)(void *interface_a, void *interface_b);
	void (*display)(starpu_data_handle handle, FILE *f);
#ifdef STARPU_USE_GORDON
	int (*convert_to_gordon)(void *interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif
	/* an identifier that is unique to each interface */
	unsigned interfaceid;
	size_t interface_size;
};

void starpu_data_register(starpu_data_handle *handleptr, uint32_t home_node,
				void *interface,
				struct starpu_data_interface_ops_t *ops);

/* "node" means memory node: 0 for main RAM, then 1, 2, etc. for various GPUs,
 * etc.
 *
 * On registration, the source of data is usually a pointer in RAM, in which
 * case 0 should be passed.
 */

void *starpu_data_get_interface_on_node(starpu_data_handle handle, unsigned memory_node);

/* Matrix interface for dense matrices */
typedef struct starpu_matrix_interface_s {
	uintptr_t ptr;
        uintptr_t dev_handle;
        size_t offset;
	uint32_t nx;
	uint32_t ny;
	uint32_t ld;
	size_t elemsize;
} starpu_matrix_interface_t;

void starpu_matrix_data_register(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t ld, uint32_t nx,
                        uint32_t ny, size_t elemsize);
uint32_t starpu_matrix_get_nx(starpu_data_handle handle);
uint32_t starpu_matrix_get_ny(starpu_data_handle handle);
uint32_t starpu_matrix_get_local_ld(starpu_data_handle handle);
uintptr_t starpu_matrix_get_local_ptr(starpu_data_handle handle);
size_t starpu_matrix_get_elemsize(starpu_data_handle handle);

/* helper methods */
#define STARPU_GET_MATRIX_PTR(interface)	(((starpu_matrix_interface_t *)(interface))->ptr)
#define STARPU_GET_MATRIX_NX(interface)	(((starpu_matrix_interface_t *)(interface))->nx)
#define STARPU_GET_MATRIX_NY(interface)	(((starpu_matrix_interface_t *)(interface))->ny)
#define STARPU_GET_MATRIX_LD(interface)	(((starpu_matrix_interface_t *)(interface))->ld)
#define STARPU_GET_MATRIX_ELEMSIZE(interface)	(((starpu_matrix_interface_t *)(interface))->elemsize)


/* BLOCK interface for 3D dense blocks */
typedef struct starpu_block_interface_s {
	uintptr_t ptr;
        uintptr_t dev_handle;
        size_t offset;
	uint32_t nx;
	uint32_t ny;
	uint32_t nz;
	uint32_t ldy;	/* number of elements between two lines */
	uint32_t ldz;	/* number of elements between two planes */
	size_t elemsize;
} starpu_block_interface_t;

void starpu_block_data_register(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t nx,
                        uint32_t ny, uint32_t nz, size_t elemsize);
uint32_t starpu_block_get_nx(starpu_data_handle handle);
uint32_t starpu_block_get_ny(starpu_data_handle handle);
uint32_t starpu_block_get_nz(starpu_data_handle handle);
uint32_t starpu_block_get_local_ldy(starpu_data_handle handle);
uint32_t starpu_block_get_local_ldz(starpu_data_handle handle);
uintptr_t starpu_block_get_local_ptr(starpu_data_handle handle);
size_t starpu_block_get_elemsize(starpu_data_handle handle);

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
        uintptr_t dev_handle;
        size_t offset;
	uint32_t nx;
	size_t elemsize;
} starpu_vector_interface_t;

void starpu_vector_data_register(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize);
uint32_t starpu_vector_get_nx(starpu_data_handle handle);
size_t starpu_vector_get_elemsize(starpu_data_handle handle);
uintptr_t starpu_vector_get_local_ptr(starpu_data_handle handle);

/* helper methods */
#define STARPU_GET_VECTOR_PTR(interface)	(((starpu_vector_interface_t *)(interface))->ptr)
#define STARPU_GET_VECTOR_NX(interface)	(((starpu_vector_interface_t *)(interface))->nx)
#define STARPU_GET_VECTOR_ELEMSIZE(interface)	(((starpu_vector_interface_t *)(interface))->elemsize)

/* variable interface for a single data (not a vector, a matrix, a list, ...) */
typedef struct starpu_variable_interface_s {
	uintptr_t ptr;
	size_t elemsize;
} starpu_variable_interface_t;

void starpu_variable_data_register(starpu_data_handle *handle, uint32_t home_node,
                        uintptr_t ptr, size_t elemsize);
size_t starpu_variable_get_elemsize(starpu_data_handle handle);
uintptr_t starpu_variable_get_local_ptr(starpu_data_handle handle);

/* helper methods */
#define STARPU_GET_VARIABLE_PTR(interface)	(((starpu_variable_interface_t *)(interface))->ptr)
#define STARPU_GET_VARIABLE_ELEMSIZE(interface)	(((starpu_variable_interface_t *)(interface))->elemsize)

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

void starpu_csr_data_register(starpu_data_handle *handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize);
uint32_t starpu_csr_get_nnz(starpu_data_handle handle);
uint32_t starpu_csr_get_nrow(starpu_data_handle handle);
uint32_t starpu_csr_get_firstentry(starpu_data_handle handle);
uintptr_t starpu_csr_get_local_nzval(starpu_data_handle handle);
uint32_t *starpu_csr_get_local_colind(starpu_data_handle handle);
uint32_t *starpu_csr_get_local_rowptr(starpu_data_handle handle);
size_t starpu_csr_get_elemsize(starpu_data_handle handle);

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

void starpu_bcsr_data_register(starpu_data_handle *handle, uint32_t home_node, uint32_t nnz, uint32_t nrow,
		uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, uint32_t r, uint32_t c, size_t elemsize);


uint32_t starpu_bcsr_get_nnz(starpu_data_handle);
uint32_t starpu_bcsr_get_nrow(starpu_data_handle);
uint32_t starpu_bcsr_get_firstentry(starpu_data_handle);
uintptr_t starpu_bcsr_get_local_nzval(starpu_data_handle);
uint32_t *starpu_bcsr_get_local_colind(starpu_data_handle);
uint32_t *starpu_bcsr_get_local_rowptr(starpu_data_handle);
uint32_t starpu_bcsr_get_r(starpu_data_handle);
uint32_t starpu_bcsr_get_c(starpu_data_handle);
size_t starpu_bcsr_get_elemsize(starpu_data_handle);

#define STARPU_MATRIX_INTERFACE_ID	0
#define STARPU_BLOCK_INTERFACE_ID	1
#define STARPU_VECTOR_INTERFACE_ID	2
#define STARPU_CSR_INTERFACE_ID		3
#define STARPU_CSC_INTERFACE_ID		4
#define STARPU_BCSCR_INTERFACE_ID	5
#define STARPU_VARIABLE_INTERFACE_ID	6
#define STARPU_NINTERFACES_ID		7 /* number of data interfaces */

unsigned starpu_get_handle_interface_id(starpu_data_handle);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_DATA_INTERFACES_H__
