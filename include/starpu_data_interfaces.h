/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Institut National de Recherche en Informatique et Automatique
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __STARPU_DATA_INTERFACES_H__
#define __STARPU_DATA_INTERFACES_H__

#include <starpu.h>
#include <starpu_data.h>
#include <starpu_util.h>

#ifdef STARPU_USE_GORDON
/* to get the gordon_strideSize_t data structure from gordon */
#include <gordon.h>
#endif

#ifdef STARPU_USE_CUDA
/* to use CUDA streams */
# ifdef STARPU_DONT_INCLUDE_CUDA_HEADERS
typedef void *cudaStream_t;
# else
#  include <cuda_runtime.h>
# endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The following structures are used to describe data interfaces */

/* This structure contains the different methods to transfer data between the
 * different types of memory nodes */
struct starpu_data_copy_methods {
	/* src type is ram */
	int (*ram_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*ram_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*ram_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*ram_to_spu)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/* src type is cuda */
	int (*cuda_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*cuda_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*cuda_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*cuda_to_spu)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/* src type is spu */
	int (*spu_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*spu_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*spu_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*spu_to_spu)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	/* src type is opencl */
	int (*opencl_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*opencl_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*opencl_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*opencl_to_spu)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

#ifdef STARPU_USE_CUDA
	/* for asynchronous CUDA transfers */
	int (*ram_to_cuda_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cudaStream_t stream);
	int (*cuda_to_ram_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cudaStream_t stream);
	int (*cuda_to_cuda_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cudaStream_t stream);
#endif

#ifdef STARPU_USE_OPENCL
	/* for asynchronous OpenCL transfers */
	/* XXX we do not use a cl_event *event type for the last argument
	 * because nvcc does not like when we have to include OpenCL headers */
        int (*ram_to_opencl_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, /* cl_event * */ void *event);
	int (*opencl_to_ram_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, /* cl_event * */ void *event);
	int (*opencl_to_opencl_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, /* cl_event * */ void *event);
#endif
};

struct starpu_data_interface_ops_t {
	/* Register an existing interface into a data handle. */
	void (*register_data_handle)(starpu_data_handle handle,
					uint32_t home_node, void *data_interface);
	/* Allocate data for the interface on a given node. */
	starpu_ssize_t (*allocate_data_on_node)(void *data_interface, uint32_t node);
	/* Free data of the interface on a given node. */
	void (*free_data_on_node)(void *data_interface, uint32_t node);
	/* ram/cuda/spu/opencl synchronous and asynchronous transfer methods */
	const struct starpu_data_copy_methods *copy_methods;
	/* Return the current pointer (if any) for the handle on the given node. */
	void * (*handle_to_pointer)(starpu_data_handle handle, uint32_t node);
	/* Return an estimation of the size of data, for performance models */
	size_t (*get_size)(starpu_data_handle handle);
	/* Return a 32bit footprint which characterizes the data size */
	uint32_t (*footprint)(starpu_data_handle handle);
	/* Compare the data size of two interfaces */
	int (*compare)(void *data_interface_a, void *data_interface_b);
	/* Dump the sizes of a handle to a file */
	void (*display)(starpu_data_handle handle, FILE *f);
#ifdef STARPU_USE_GORDON
	/* Convert the data size to the spu size format */
	int (*convert_to_gordon)(void *data_interface, uint64_t *ptr, gordon_strideSize_t *ss); 
#endif
	/* an identifier that is unique to each interface */
	unsigned interfaceid;
	/* The size of the interface data descriptor */
	size_t interface_size;
};

void starpu_data_register(starpu_data_handle *handleptr, uint32_t home_node,
				void *data_interface,
				struct starpu_data_interface_ops_t *ops);

/* Return the pointer associated with HANDLE on node NODE or NULL if HANDLE's
 * interface does not support this operation or data for this handle is not
 * allocated on that node. */
void *starpu_handle_to_pointer(starpu_data_handle handle, uint32_t node);

/* Return the local pointer associated with HANDLE or NULL if HANDLE's
 * interface does not have data allocated locally */
void *starpu_handle_get_local_ptr(starpu_data_handle handle);

extern struct starpu_data_interface_ops_t _starpu_interface_matrix_ops;

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
#define STARPU_MATRIX_GET_PTR(interface)	(((starpu_matrix_interface_t *)(interface))->ptr)
#define STARPU_MATRIX_GET_NX(interface)	(((starpu_matrix_interface_t *)(interface))->nx)
#define STARPU_MATRIX_GET_NY(interface)	(((starpu_matrix_interface_t *)(interface))->ny)
#define STARPU_MATRIX_GET_LD(interface)	(((starpu_matrix_interface_t *)(interface))->ld)
#define STARPU_MATRIX_GET_ELEMSIZE(interface)	(((starpu_matrix_interface_t *)(interface))->elemsize)


/* BLOCK interface for 3D dense blocks */
/* TODO: rename to 3dmatrix? */
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
#define STARPU_BLOCK_GET_PTR(interface)	(((starpu_block_interface_t *)(interface))->ptr)
#define STARPU_BLOCK_GET_NX(interface)	(((starpu_block_interface_t *)(interface))->nx)
#define STARPU_BLOCK_GET_NY(interface)	(((starpu_block_interface_t *)(interface))->ny)
#define STARPU_BLOCK_GET_NZ(interface)	(((starpu_block_interface_t *)(interface))->nz)
#define STARPU_BLOCK_GET_LDY(interface)	(((starpu_block_interface_t *)(interface))->ldy)
#define STARPU_BLOCK_GET_LDZ(interface)	(((starpu_block_interface_t *)(interface))->ldz)
#define STARPU_BLOCK_GET_ELEMSIZE(interface)	(((starpu_block_interface_t *)(interface))->elemsize)

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
#define STARPU_VECTOR_GET_PTR(interface)	(((starpu_vector_interface_t *)(interface))->ptr)
#define STARPU_VECTOR_GET_NX(interface)	(((starpu_vector_interface_t *)(interface))->nx)
#define STARPU_VECTOR_GET_ELEMSIZE(interface)	(((starpu_vector_interface_t *)(interface))->elemsize)

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
#define STARPU_VARIABLE_GET_PTR(interface)	(((starpu_variable_interface_t *)(interface))->ptr)
#define STARPU_VARIABLE_GET_ELEMSIZE(interface)	(((starpu_variable_interface_t *)(interface))->elemsize)

/* void interface. There is no data really associated to that interface, but it
 * may be used as a synchronization mechanism. It also permits to express an
 * abstract piece of data that is managed by the application internally: this
 * makes it possible to forbid the concurrent execution of different tasks
 * accessing the same "void" data in read-write concurrently. */
void starpu_void_data_register(starpu_data_handle *handleptr);

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

#define STARPU_CSR_GET_NNZ(interface)	(((starpu_csr_interface_t *)(interface))->nnz)
#define STARPU_CSR_GET_NROW(interface)	(((starpu_csr_interface_t *)(interface))->nrow)
#define STARPU_CSR_GET_NZVAL(interface)	(((starpu_csr_interface_t *)(interface))->nzval)
#define STARPU_CSR_GET_COLIND(interface)	(((starpu_csr_interface_t *)(interface))->colind)
#define STARPU_CSR_GET_ROWPTR(interface)	(((starpu_csr_interface_t *)(interface))->rowptr)
#define STARPU_CSR_GET_FIRSTENTRY(interface)	(((starpu_csr_interface_t *)(interface))->firstentry)
#define STARPU_CSR_GET_ELEMSIZE(interface)	(((starpu_csr_interface_t *)(interface))->elemsize)

/* BCSR interface for sparse matrices (blocked compressed sparse row
 * representation) */
typedef struct starpu_bcsr_interface_s {
	uint32_t nnz; /* number of non-zero BLOCKS */
	uint32_t nrow; /* number of rows (in terms of BLOCKS) */

	uintptr_t nzval; /* non-zero values */
	uint32_t *colind; /* position of non-zero entried on the row */
/*	uint32_t *rowind; */ /* position of non-zero entried on the col */
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

/*
 * Multiformat interface
 */
struct starpu_multiformat_data_interface_ops {
	size_t cpu_elemsize;
#ifdef STARPU_USE_OPENCL
	size_t opencl_elemsize;
	struct starpu_codelet_t *cpu_to_opencl_cl;
	struct starpu_codelet_t *opencl_to_cpu_cl;
#endif
#ifdef STARPU_USE_CUDA
	size_t cuda_elemsize;
	struct starpu_codelet_t *cpu_to_cuda_cl;
	struct starpu_codelet_t *cuda_to_cpu_cl;
#endif
};

typedef struct starpu_multiformat_interface_s {
	void *cpu_ptr;
#ifdef STARPU_USE_CUDA
	void *cuda_ptr;
#endif
#ifdef STARPU_USE_OPENCL
	void *opencl_ptr;
#endif
	uintptr_t dev_handle;
	size_t offset;
	uint32_t nx;
	struct starpu_multiformat_data_interface_ops *ops;
	double conversion_time;
} starpu_multiformat_interface_t;

void starpu_multiformat_data_register(starpu_data_handle *handle,
				      uint32_t home_node,
				      void *ptr,
				      uint32_t nobjects,
				      struct starpu_multiformat_data_interface_ops *format_ops);

#define STARPU_MULTIFORMAT_GET_PTR(interface)  (((starpu_multiformat_interface_t *)(interface))->cpu_ptr)

#ifdef STARPU_USE_CUDA
#define STARPU_MULTIFORMAT_GET_CUDA_PTR(interface) (((starpu_multiformat_interface_t *)(interface))->cuda_ptr)
#endif

#ifdef STARPU_USE_OPENCL
#define STARPU_MULTIFORMAT_GET_OPENCL_PTR(interface) (((starpu_multiformat_interface_t *)(interface))->opencl_ptr)
#endif

#define STARPU_MULTIFORMAT_GET_NX(interface)  (((starpu_multiformat_interface_t *)(interface))->nx)

#define STARPU_MATRIX_INTERFACE_ID	0
#define STARPU_BLOCK_INTERFACE_ID	1
#define STARPU_VECTOR_INTERFACE_ID	2
#define STARPU_CSR_INTERFACE_ID		3
#define STARPU_BCSR_INTERFACE_ID	4
#define STARPU_VARIABLE_INTERFACE_ID	5
#define STARPU_VOID_INTERFACE_ID	6
#define STARPU_MULTIFORMAT_INTERFACE_ID 7
#define STARPU_NINTERFACES_ID		8 /* number of data interfaces */

unsigned starpu_get_handle_interface_id(starpu_data_handle);

/* Lookup a ram pointer into a StarPU handle */
extern starpu_data_handle starpu_data_lookup(const void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DATA_INTERFACES_H__ */
