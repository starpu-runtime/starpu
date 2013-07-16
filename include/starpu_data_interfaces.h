/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011-2012  Institut National de Recherche en Informatique et Automatique
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

#ifdef STARPU_USE_CUDA
/* to use CUDA streams */
# ifdef STARPU_DONT_INCLUDE_CUDA_HEADERS
typedef void *cudaStream_t;
# else
#  include <cuda_runtime.h>
# endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_data_copy_methods
{
	int (*ram_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*ram_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*ram_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*ram_to_mic)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	int (*cuda_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*cuda_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*cuda_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	int (*opencl_to_ram)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*opencl_to_cuda)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*opencl_to_opencl)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

	int (*mic_to_ram)(void *src_interface, unsigned srd_node, void *dst_interface, unsigned dst_node);

	int (*scc_src_to_sink)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*scc_sink_to_src)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*scc_sink_to_sink)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node);

#ifdef STARPU_USE_CUDA
	int (*ram_to_cuda_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cudaStream_t stream);
	int (*cuda_to_ram_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cudaStream_t stream);
	int (*cuda_to_cuda_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cudaStream_t stream);
#else
#ifdef STARPU_SIMGRID
	int cuda_to_cuda_async;
#endif
#endif

#if defined(STARPU_USE_OPENCL) && !defined(__CUDACC__)
	int (*ram_to_opencl_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event);
	int (*opencl_to_ram_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event);
	int (*opencl_to_opencl_async)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, cl_event *event);
#endif

#ifdef STARPU_USE_MIC
	int (*ram_to_mic_async)(void *src_intreface, unsigned src_node, void *dst_interface, unsigned dst_node);
	int (*mic_to_ram_async)(void *src_interface, unsigned srd_node, void *dst_interface, unsigned dst_node);
#endif

	int (*any_to_any)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);
};

int starpu_interface_copy(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, void *async_data);
uintptr_t starpu_malloc_on_node(unsigned dst_node, size_t size);
void starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size);

enum starpu_data_interface_id
{
	STARPU_UNKNOWN_INTERFACE_ID = -1,
	STARPU_MATRIX_INTERFACE_ID=0,
	STARPU_BLOCK_INTERFACE_ID=1,
	STARPU_VECTOR_INTERFACE_ID=2,
	STARPU_CSR_INTERFACE_ID=3,
	STARPU_BCSR_INTERFACE_ID=4,
	STARPU_VARIABLE_INTERFACE_ID=5,
	STARPU_VOID_INTERFACE_ID=6,
	STARPU_MULTIFORMAT_INTERFACE_ID=7,
	STARPU_COO_INTERFACE_ID=8,
	STARPU_MAX_INTERFACE_ID=9 /* maximum number of data interfaces */
};

struct starpu_data_interface_ops
{
	void		 (*register_data_handle)	(starpu_data_handle_t handle,
								unsigned home_node, void *data_interface);
	starpu_ssize_t	 (*allocate_data_on_node)	(void *data_interface, unsigned node);
	void 		 (*free_data_on_node)		(void *data_interface, unsigned node);
	const struct starpu_data_copy_methods *copy_methods;
	void * 		 (*handle_to_pointer)		(starpu_data_handle_t handle, unsigned node);
	size_t 		 (*get_size)			(starpu_data_handle_t handle);
	uint32_t 	 (*footprint)			(starpu_data_handle_t handle);
	int 		 (*compare)			(void *data_interface_a, void *data_interface_b);
	void 		 (*display)			(starpu_data_handle_t handle, FILE *f);
	enum starpu_data_interface_id interfaceid;
	size_t interface_size;

	int is_multiformat;
	struct starpu_multiformat_data_interface_ops* (*get_mf_ops)(void *data_interface);

	int (*pack_data) (starpu_data_handle_t handle, unsigned node, void **ptr, starpu_ssize_t *count);
	int (*unpack_data) (starpu_data_handle_t handle, unsigned node, void *ptr, size_t count);
};

int starpu_data_interface_get_next_id(void);

void starpu_data_register(starpu_data_handle_t *handleptr, unsigned home_node, void *data_interface, struct starpu_data_interface_ops *ops);
void starpu_data_register_same(starpu_data_handle_t *handledst, starpu_data_handle_t handlesrc);

void *starpu_data_handle_to_pointer(starpu_data_handle_t handle, unsigned node);
void *starpu_data_get_local_ptr(starpu_data_handle_t handle);

void *starpu_data_get_interface_on_node(starpu_data_handle_t handle, unsigned memory_node);

extern struct starpu_data_interface_ops starpu_interface_matrix_ops;

struct starpu_matrix_interface
{
	enum starpu_data_interface_id id;

	uintptr_t ptr;
	uintptr_t dev_handle;
	size_t offset;
	uint32_t nx;
	uint32_t ny;
	uint32_t ld;
	size_t elemsize;
};

void starpu_matrix_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr, uint32_t ld, uint32_t nx, uint32_t ny, size_t elemsize);
uint32_t starpu_matrix_get_nx(starpu_data_handle_t handle);
uint32_t starpu_matrix_get_ny(starpu_data_handle_t handle);
uint32_t starpu_matrix_get_local_ld(starpu_data_handle_t handle);
uintptr_t starpu_matrix_get_local_ptr(starpu_data_handle_t handle);
size_t starpu_matrix_get_elemsize(starpu_data_handle_t handle);

#define STARPU_MATRIX_GET_PTR(interface)	(((struct starpu_matrix_interface *)(interface))->ptr)
#define STARPU_MATRIX_GET_DEV_HANDLE(interface)	(((struct starpu_matrix_interface *)(interface))->dev_handle)
#define STARPU_MATRIX_GET_OFFSET(interface)	(((struct starpu_matrix_interface *)(interface))->offset)
#define STARPU_MATRIX_GET_NX(interface)	(((struct starpu_matrix_interface *)(interface))->nx)
#define STARPU_MATRIX_GET_NY(interface)	(((struct starpu_matrix_interface *)(interface))->ny)
#define STARPU_MATRIX_GET_LD(interface)	(((struct starpu_matrix_interface *)(interface))->ld)
#define STARPU_MATRIX_GET_ELEMSIZE(interface)	(((struct starpu_matrix_interface *)(interface))->elemsize)

struct starpu_coo_interface
{
	enum starpu_data_interface_id id;

	uint32_t  *columns;
	uint32_t  *rows;
	uintptr_t values;
	uint32_t  nx;
	uint32_t  ny;
	uint32_t  n_values;
	size_t    elemsize;
};

void starpu_coo_data_register(starpu_data_handle_t *handleptr, unsigned home_node, uint32_t nx, uint32_t ny, uint32_t n_values, uint32_t *columns, uint32_t *rows, uintptr_t values, size_t elemsize);

#define STARPU_COO_GET_COLUMNS(interface) \
	(((struct starpu_coo_interface *)(interface))->columns)
#define STARPU_COO_GET_COLUMNS_DEV_HANDLE(interface) \
	(((struct starpu_coo_interface *)(interface))->columns)
#define STARPU_COO_GET_ROWS(interface) \
	(((struct starpu_coo_interface *)(interface))->rows)
#define STARPU_COO_GET_ROWS_DEV_HANDLE(interface) \
	(((struct starpu_coo_interface *)(interface))->rows)
#define STARPU_COO_GET_VALUES(interface) \
	(((struct starpu_coo_interface *)(interface))->values)
#define STARPU_COO_GET_VALUES_DEV_HANDLE(interface) \
	(((struct starpu_coo_interface *)(interface))->values)
#define STARPU_COO_GET_OFFSET 0
#define STARPU_COO_GET_NX(interface) \
	(((struct starpu_coo_interface *)(interface))->nx)
#define STARPU_COO_GET_NY(interface) \
	(((struct starpu_coo_interface *)(interface))->ny)
#define STARPU_COO_GET_NVALUES(interface) \
	(((struct starpu_coo_interface *)(interface))->n_values)
#define STARPU_COO_GET_ELEMSIZE(interface) \
	(((struct starpu_coo_interface *)(interface))->elemsize)

/* TODO: rename to 3dmatrix? */
struct starpu_block_interface
{
	enum starpu_data_interface_id id;

	uintptr_t ptr;
	uintptr_t dev_handle;
	size_t offset;
	uint32_t nx;
	uint32_t ny;
	uint32_t nz;
	uint32_t ldy;
	uint32_t ldz;
	size_t elemsize;
};

void starpu_block_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t nx, uint32_t ny, uint32_t nz, size_t elemsize);
uint32_t starpu_block_get_nx(starpu_data_handle_t handle);
uint32_t starpu_block_get_ny(starpu_data_handle_t handle);
uint32_t starpu_block_get_nz(starpu_data_handle_t handle);
uint32_t starpu_block_get_local_ldy(starpu_data_handle_t handle);
uint32_t starpu_block_get_local_ldz(starpu_data_handle_t handle);
uintptr_t starpu_block_get_local_ptr(starpu_data_handle_t handle);
size_t starpu_block_get_elemsize(starpu_data_handle_t handle);

#define STARPU_BLOCK_GET_PTR(interface)	(((struct starpu_block_interface *)(interface))->ptr)
#define STARPU_BLOCK_GET_DEV_HANDLE(interface)	(((struct starpu_block_interface *)(interface))->dev_handle)
#define STARPU_BLOCK_GET_OFFSET(interface)	(((struct starpu_block_interface *)(interface))->offset)
#define STARPU_BLOCK_GET_NX(interface)	(((struct starpu_block_interface *)(interface))->nx)
#define STARPU_BLOCK_GET_NY(interface)	(((struct starpu_block_interface *)(interface))->ny)
#define STARPU_BLOCK_GET_NZ(interface)	(((struct starpu_block_interface *)(interface))->nz)
#define STARPU_BLOCK_GET_LDY(interface)	(((struct starpu_block_interface *)(interface))->ldy)
#define STARPU_BLOCK_GET_LDZ(interface)	(((struct starpu_block_interface *)(interface))->ldz)
#define STARPU_BLOCK_GET_ELEMSIZE(interface)	(((struct starpu_block_interface *)(interface))->elemsize)

struct starpu_vector_interface
{
	enum starpu_data_interface_id id;

	uintptr_t ptr;
	uintptr_t dev_handle;
	size_t offset;
	uint32_t nx;
	size_t elemsize;
};

void starpu_vector_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr, uint32_t nx, size_t elemsize);
uint32_t starpu_vector_get_nx(starpu_data_handle_t handle);
size_t starpu_vector_get_elemsize(starpu_data_handle_t handle);
uintptr_t starpu_vector_get_local_ptr(starpu_data_handle_t handle);

#define STARPU_VECTOR_GET_PTR(interface)	(((struct starpu_vector_interface *)(interface))->ptr)
#define STARPU_VECTOR_GET_DEV_HANDLE(interface)	(((struct starpu_vector_interface *)(interface))->dev_handle)
#define STARPU_VECTOR_GET_OFFSET(interface)	(((struct starpu_vector_interface *)(interface))->offset)
#define STARPU_VECTOR_GET_NX(interface)	(((struct starpu_vector_interface *)(interface))->nx)
#define STARPU_VECTOR_GET_ELEMSIZE(interface)	(((struct starpu_vector_interface *)(interface))->elemsize)

struct starpu_variable_interface
{
	enum starpu_data_interface_id id;

	uintptr_t ptr;
	uintptr_t dev_handle;
	size_t offset;
	size_t elemsize;
};

void starpu_variable_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr, size_t size);
size_t starpu_variable_get_elemsize(starpu_data_handle_t handle);
uintptr_t starpu_variable_get_local_ptr(starpu_data_handle_t handle);

#define STARPU_VARIABLE_GET_PTR(interface)	(((struct starpu_variable_interface *)(interface))->ptr)
#define STARPU_VARIABLE_GET_OFFSET(interface)	(((struct starpu_variable_interface *)(interface))->offset)
#define STARPU_VARIABLE_GET_ELEMSIZE(interface)	(((struct starpu_variable_interface *)(interface))->elemsize)
#define STARPU_VARIABLE_GET_DEV_HANDLE(interface) \
	(((struct starpu_variable_interface *)(interface))->ptr)

void starpu_void_data_register(starpu_data_handle_t *handle);

struct starpu_csr_interface
{
	enum starpu_data_interface_id id;

	uint32_t nnz;
	uint32_t nrow;
	uintptr_t nzval;
	uint32_t *colind;
	uint32_t *rowptr;

	uint32_t firstentry;

	size_t elemsize;
};

void starpu_csr_data_register(starpu_data_handle_t *handle, unsigned home_node, uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, size_t elemsize);
uint32_t starpu_csr_get_nnz(starpu_data_handle_t handle);
uint32_t starpu_csr_get_nrow(starpu_data_handle_t handle);
uint32_t starpu_csr_get_firstentry(starpu_data_handle_t handle);
uintptr_t starpu_csr_get_local_nzval(starpu_data_handle_t handle);
uint32_t *starpu_csr_get_local_colind(starpu_data_handle_t handle);
uint32_t *starpu_csr_get_local_rowptr(starpu_data_handle_t handle);
size_t starpu_csr_get_elemsize(starpu_data_handle_t handle);

#define STARPU_CSR_GET_NNZ(interface)	(((struct starpu_csr_interface *)(interface))->nnz)
#define STARPU_CSR_GET_NROW(interface)	(((struct starpu_csr_interface *)(interface))->nrow)
#define STARPU_CSR_GET_NZVAL(interface)	(((struct starpu_csr_interface *)(interface))->nzval)
#define STARPU_CSR_GET_NZVAL_DEV_HANDLE(interface)		\
	(((struct starpu_csr_interface *)(interface))->nnz)
#define STARPU_CSR_GET_COLIND(interface)	(((struct starpu_csr_interface *)(interface))->colind)
#define STARPU_CSR_GET_COLIND_DEV_HANDLE(interface) \
	(((struct starpu_csr_interface *)(interface))->colind)
#define STARPU_CSR_GET_ROWPTR(interface)	(((struct starpu_csr_interface *)(interface))->rowptr)
#define STARPU_CSR_GET_ROWPTR_DEV_HANDLE(interface)		\
	(((struct starpu_csr_interface *)(interface))->rowptr)
#define STARPU_CSR_GET_OFFSET 0
#define STARPU_CSR_GET_FIRSTENTRY(interface)	(((struct starpu_csr_interface *)(interface))->firstentry)
#define STARPU_CSR_GET_ELEMSIZE(interface)	(((struct starpu_csr_interface *)(interface))->elemsize)

struct starpu_bcsr_interface
{
	enum starpu_data_interface_id id;

	uint32_t nnz;
	uint32_t nrow;

	uintptr_t nzval;
	uint32_t *colind;
	uint32_t *rowptr;

	uint32_t firstentry;

	uint32_t r;
	uint32_t c;

	size_t elemsize;
};

void starpu_bcsr_data_register(starpu_data_handle_t *handle, unsigned home_node, uint32_t nnz, uint32_t nrow, uintptr_t nzval, uint32_t *colind, uint32_t *rowptr, uint32_t firstentry, uint32_t r, uint32_t c, size_t elemsize);

#define STARPU_BCSR_GET_NNZ(interface)        (((struct starpu_bcsr_interface *)(interface))->nnz)
#define STARPU_BCSR_GET_NZVAL(interface)      (((struct starpu_bcsr_interface *)(interface))->nzval)
#define STARPU_BCSR_GET_NZVAL_DEV_HANDLE(interface) \
	(((struct starpu_bcsr_interface *)(interface))->nnz)
#define STARPU_BCSR_GET_COLIND(interface)     (((struct starpu_bcsr_interface *)(interface))->colind)
#define STARPU_BCSR_GET_COLIND_DEV_HANDLE(interface) \
	(((struct starpu_bcsr_interface *)(interface))->colind)
#define STARPU_BCSR_GET_ROWPTR(interface)     (((struct starpu_bcsr_interface *)(interface))->rowptr)
#define STARPU_BCSR_GET_ROWPTR_DEV_HANDLE(interface) \
	(((struct starpu_bcsr_interface *)(interface))->rowptr)
#define STARPU_BCSR_GET_OFFSET 0
uint32_t starpu_bcsr_get_nnz(starpu_data_handle_t handle);
uint32_t starpu_bcsr_get_nrow(starpu_data_handle_t handle);
uint32_t starpu_bcsr_get_firstentry(starpu_data_handle_t handle);
uintptr_t starpu_bcsr_get_local_nzval(starpu_data_handle_t handle);
uint32_t *starpu_bcsr_get_local_colind(starpu_data_handle_t handle);
uint32_t *starpu_bcsr_get_local_rowptr(starpu_data_handle_t handle);
uint32_t starpu_bcsr_get_r(starpu_data_handle_t handle);
uint32_t starpu_bcsr_get_c(starpu_data_handle_t handle);
size_t starpu_bcsr_get_elemsize(starpu_data_handle_t handle);

struct starpu_multiformat_data_interface_ops
{
	size_t cpu_elemsize;
	size_t opencl_elemsize;
	struct starpu_codelet *cpu_to_opencl_cl;
	struct starpu_codelet *opencl_to_cpu_cl;
	size_t cuda_elemsize;
	struct starpu_codelet *cpu_to_cuda_cl;
	struct starpu_codelet *cuda_to_cpu_cl;
	size_t mic_elemsize;
	struct starpu_codelet *cpu_to_mic_cl;
	struct starpu_codelet *mic_to_cpu_cl;
};

struct starpu_multiformat_interface
{
	enum starpu_data_interface_id id;

	void *cpu_ptr;
	void *cuda_ptr;
	void *opencl_ptr;
	void *mic_ptr;
	uint32_t nx;
	struct starpu_multiformat_data_interface_ops *ops;
};

void starpu_multiformat_data_register(starpu_data_handle_t *handle, unsigned home_node, void *ptr, uint32_t nobjects, struct starpu_multiformat_data_interface_ops *format_ops);

#define STARPU_MULTIFORMAT_GET_CPU_PTR(interface)  (((struct starpu_multiformat_interface *)(interface))->cpu_ptr)
#define STARPU_MULTIFORMAT_GET_CUDA_PTR(interface) (((struct starpu_multiformat_interface *)(interface))->cuda_ptr)
#define STARPU_MULTIFORMAT_GET_OPENCL_PTR(interface) (((struct starpu_multiformat_interface *)(interface))->opencl_ptr)
#define STARPU_MULTIFORMAT_GET_MIC_PTR(interface) (((struct starpu_multiformat_interface *)(interface))->mic_ptr)
#define STARPU_MULTIFORMAT_GET_NX(interface)  (((struct starpu_multiformat_interface *)(interface))->nx)

enum starpu_data_interface_id starpu_data_get_interface_id(starpu_data_handle_t handle);

int starpu_data_pack(starpu_data_handle_t handle, void **ptr, starpu_ssize_t *count);
int starpu_data_unpack(starpu_data_handle_t handle, void *ptr, size_t count);
size_t starpu_data_get_size(starpu_data_handle_t handle);

starpu_data_handle_t starpu_data_lookup(const void *ptr);

struct starpu_disk_interface
{
	uintptr_t dev_handle;
};

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_DATA_INTERFACES_H__ */
