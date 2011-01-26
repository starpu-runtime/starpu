/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2011 (see AUTHORS file)
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

/*
 * Conjugate gradients for Sparse matrices
 */

#include "dw_spmv.h"

#ifdef STARPU_USE_CUDA
extern void spmv_kernel_cuda(void *descr[], void *args);
#endif

struct timeval start;
struct timeval end;

#ifdef STARPU_USE_OPENCL
#include "starpu_opencl.h"
struct starpu_opencl_program opencl_codelet;
void spmv_kernel_opencl(void *descr[], void *args)
{
	cl_kernel kernel;
	cl_command_queue queue;
	cl_event event;
	int id, devid, err, n;

	uint32_t nnz = STARPU_CSR_GET_NNZ(descr[0]);
	uint32_t nrow = STARPU_CSR_GET_NROW(descr[0]);
	float *nzval = (float *)STARPU_CSR_GET_NZVAL(descr[0]);
	uint32_t *colind = STARPU_CSR_GET_COLIND(descr[0]);
	uint32_t *rowptr = STARPU_CSR_GET_ROWPTR(descr[0]);
	uint32_t firstentry = STARPU_CSR_GET_FIRSTENTRY(descr[0]);

	float *vecin = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	uint32_t nx_in = STARPU_VECTOR_GET_NX(descr[1]);

	float *vecout = (float *)STARPU_VECTOR_GET_PTR(descr[2]);
	uint32_t nx_out = STARPU_VECTOR_GET_NX(descr[2]);

        id = starpu_worker_get_id();
        devid = starpu_worker_get_devid(id);

        err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_codelet, "spvm", devid);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = 0;
        n=0;
	err = clSetKernelArg(kernel, n++, sizeof(uint32_t), &nnz);
	err = clSetKernelArg(kernel, n++, sizeof(uint32_t), &nrow);
	err = clSetKernelArg(kernel, n++, sizeof(cl_mem), &nzval);
	err = clSetKernelArg(kernel, n++, sizeof(cl_mem), &colind);
	err = clSetKernelArg(kernel, n++, sizeof(cl_mem), &rowptr);
	err = clSetKernelArg(kernel, n++, sizeof(uint32_t), &firstentry);
	err = clSetKernelArg(kernel, n++, sizeof(cl_mem), &vecin);
	err = clSetKernelArg(kernel, n++, sizeof(uint32_t), &nx_in);
	err = clSetKernelArg(kernel, n++, sizeof(cl_mem), &vecout);
	err = clSetKernelArg(kernel, n++, sizeof(uint32_t), &nx_out);
        if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
                size_t global=1024;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, &event);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	clFinish(queue);
	starpu_opencl_collect_stats(event);
	clReleaseEvent(event);

        starpu_opencl_release_kernel(kernel);
}
#endif

unsigned nblocks = 2;
uint32_t size = 4194304;

starpu_data_handle sparse_matrix;
starpu_data_handle vector_in, vector_out;

float *sparse_matrix_nzval;
uint32_t *sparse_matrix_colind;
uint32_t *sparse_matrix_rowptr;

float *vector_in_ptr;
float *vector_out_ptr;

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}
	}
}

static void cpu_spmv(void *descr[], __attribute__((unused))  void *arg)
{
	float *nzval = (float *)STARPU_CSR_GET_NZVAL(descr[0]);
	uint32_t *colind = STARPU_CSR_GET_COLIND(descr[0]);
	uint32_t *rowptr = STARPU_CSR_GET_ROWPTR(descr[0]);

	float *vecin = (float *)STARPU_VECTOR_GET_PTR(descr[1]);
	float *vecout = (float *)STARPU_VECTOR_GET_PTR(descr[2]);

	uint32_t firstelem = STARPU_CSR_GET_FIRSTENTRY(descr[0]);

	uint32_t nnz;
	uint32_t nrow;

	nnz = STARPU_CSR_GET_NNZ(descr[0]);
	nrow = STARPU_CSR_GET_NROW(descr[0]);

	//STARPU_ASSERT(nrow == STARPU_VECTOR_GET_NX(descr[1]));
	STARPU_ASSERT(nrow == STARPU_VECTOR_GET_NX(descr[2]));

	unsigned row;
	for (row = 0; row < nrow; row++)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstelem;
		unsigned lastindex = rowptr[row+1] - firstelem;

		for (index = firstindex; index < lastindex; index++)
		{
			unsigned col;

			col = colind[index];
			tmp += nzval[index]*vecin[col];
		}

		vecout[row] = tmp;
	}

}

static void create_data(void)
{
	/* we need a sparse symetric (definite positive ?) matrix and a "dense" vector */
	
	/* example of 3-band matrix */
	float *nzval;
	uint32_t nnz;
	uint32_t *colind;
	uint32_t *rowptr;

	nnz = 3*size-2;

	nzval = malloc(nnz*sizeof(float));
	colind = malloc(nnz*sizeof(uint32_t));
	rowptr = malloc((size+1)*sizeof(uint32_t));

	assert(nzval);
	assert(colind);
	assert(rowptr);

	/* fill the matrix */
	unsigned row;
	unsigned pos = 0;
	for (row = 0; row < size; row++)
	{
		rowptr[row] = pos;

		if (row > 0) {
			nzval[pos] = 1.0f;
			colind[pos] = row-1;
			pos++;
		}
		
		nzval[pos] = 5.0f;
		colind[pos] = row;
		pos++;

		if (row < size - 1) {
			nzval[pos] = 1.0f;
			colind[pos] = row+1;
			pos++;
		}
	}

	STARPU_ASSERT(pos == nnz);

	rowptr[size] = nnz;
	
	starpu_csr_data_register(&sparse_matrix, 0, nnz, size, (uintptr_t)nzval, colind, rowptr, 0, sizeof(float));

	sparse_matrix_nzval = nzval;
	sparse_matrix_colind = colind;
	sparse_matrix_rowptr = rowptr;

	/* initiate the 2 vectors */
	float *invec, *outvec;
	invec = malloc(size*sizeof(float));
	assert(invec);

	outvec = malloc(size*sizeof(float));
	assert(outvec);

	/* fill those */
	unsigned ind;
	for (ind = 0; ind < size; ind++)
	{
		invec[ind] = 2.0f;
		outvec[ind] = 0.0f;
	}

	starpu_vector_data_register(&vector_in, 0, (uintptr_t)invec, size, sizeof(float));
	starpu_vector_data_register(&vector_out, 0, (uintptr_t)outvec, size, sizeof(float));

	vector_in_ptr = invec;
	vector_out_ptr = outvec;

}

void call_spmv_codelet_filters(void)
{

	/* partition the data along a block distribution */
	struct starpu_data_filter csr_f, vector_f;
	csr_f.filter_func = starpu_vertical_block_filter_func_csr;
	csr_f.nchildren = nblocks;
	csr_f.get_nchildren = NULL;
	/* the children also use a csr interface */
	csr_f.get_child_ops = NULL;

	vector_f.filter_func = starpu_block_filter_func_vector;
	vector_f.nchildren = nblocks;
	vector_f.get_nchildren = NULL;
	vector_f.get_child_ops = NULL;

	starpu_data_partition(sparse_matrix, &csr_f);
	starpu_data_partition(vector_out, &vector_f);

#ifdef STARPU_USE_OPENCL
        {
                int ret = starpu_opencl_load_opencl_from_file("examples/spmv/spmv_opencl.cl", &opencl_codelet);
                if (ret)
		{
			fprintf(stderr, "Failed to compile OpenCL codelet\n");
			exit(ret);
		}
        }
#endif

	starpu_codelet cl = {};

	cl.where = STARPU_CPU|STARPU_CUDA|STARPU_OPENCL;
	cl.cpu_func =  cpu_spmv;
#ifdef STARPU_USE_CUDA
	cl.cuda_func = spmv_kernel_cuda;
#endif
#ifdef STARPU_USE_OPENCL
        cl.opencl_func = spmv_kernel_opencl;
#endif
	cl.nbuffers = 3;
	cl.model = NULL;

	gettimeofday(&start, NULL);

	unsigned part;
	for (part = 0; part < nblocks; part++)
	{
		struct starpu_task *task = starpu_task_create();
                int ret;

		task->callback_func = NULL;

		task->cl = &cl;
		task->cl_arg = NULL;
	
		task->buffers[0].handle = starpu_data_get_sub_data(sparse_matrix, 1, part);
		task->buffers[0].mode  = STARPU_R;
		task->buffers[1].handle = vector_in;
		task->buffers[1].mode = STARPU_R;
		task->buffers[2].handle = starpu_data_get_sub_data(vector_out, 1, part);
		task->buffers[2].mode = STARPU_W;
	
		ret = starpu_task_submit(task);
		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			fprintf(stderr, "No worker may execute this task\n");
			exit(0);
		}
	}

	starpu_task_wait_for_all();

	gettimeofday(&end, NULL);

	starpu_data_unpartition(sparse_matrix, 0);
	starpu_data_unpartition(vector_out, 0);
}

static void print_results(void)
{
	unsigned row;

	for (row = 0; row < STARPU_MIN(size, 16); row++)
	{
		printf("%2.2f\t%2.2f\n", vector_in_ptr[row], vector_out_ptr[row]);
	}
}

int main(__attribute__ ((unused)) int argc,
	__attribute__ ((unused)) char **argv)
{
	parse_args(argc, argv);

	/* start the runtime */
	starpu_init(NULL);

	/* create the sparse input matrix */
	create_data();

	/* create a new codelet that will perform a SpMV on it */
	call_spmv_codelet_filters();

	starpu_shutdown();

	print_results();

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

	return 0;
}
