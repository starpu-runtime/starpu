/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include "../helper.h"
#include <sys/time.h>

#ifdef STARPU_USE_CUDA
#  include <cublas.h>
#endif

#define LOOPS 100

void vector_cpu_func(void *descr[], void *cl_arg __attribute__((unused)))
{
	STARPU_SKIP_IF_VALGRIND;

	float *matrix = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	int nx = STARPU_VECTOR_GET_NX(descr[0]);
	int i;
	float sum=0;

	for(i=0 ; i<nx ; i++) sum+=i;
	matrix[0] = sum/nx;
}

void vector_cuda_func(void *descr[], void *cl_arg __attribute__((unused)))
{
#ifdef STARPU_USE_CUDA
	STARPU_SKIP_IF_VALGRIND;

	float *matrix = (float *)STARPU_VECTOR_GET_PTR(descr[0]);
	int nx = STARPU_VECTOR_GET_NX(descr[0]);

	float sum = cublasSasum(nx, matrix, 1);
	cudaThreadSynchronize();
	sum /= nx;

	cudaMemcpy(matrix, &sum, sizeof(matrix[0]), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
#endif /* STARPU_USE_CUDA */
}

void matrix_cpu_func(void *descr[], void *cl_arg __attribute__((unused)))
{
	STARPU_SKIP_IF_VALGRIND;

	float *matrix = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	int nx = STARPU_MATRIX_GET_NX(descr[0]);
	int ny = STARPU_MATRIX_GET_NY(descr[0]);
	int i;
	float sum=0;

	for(i=0 ; i<nx*ny ; i++) sum+=i;
	matrix[0] = sum / (nx*ny);
}

void matrix_cuda_func(void *descr[], void *cl_arg __attribute__((unused)))
{
#ifdef STARPU_USE_CUDA
	STARPU_SKIP_IF_VALGRIND;

	float *matrix = (float *)STARPU_MATRIX_GET_PTR(descr[0]);
	int nx = STARPU_MATRIX_GET_NX(descr[0]);
	int ny = STARPU_MATRIX_GET_NY(descr[0]);

	float sum = cublasSasum(nx*ny, matrix, 1);
	cudaThreadSynchronize();
	sum /= nx*ny;

	cudaMemcpy(matrix, &sum, sizeof(matrix[0]), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
#endif /* STARPU_USE_CUDA */
}

int check_size(int nx, struct starpu_codelet *vector_codelet, struct starpu_codelet *matrix_codelet, char *device_name)
{
	float *matrix, mean;
	starpu_data_handle_t vector_handle, matrix_handle;
	int ret, i, loop;
	double vector_timing, matrix_timing;
	struct timeval start;
	struct timeval end;

	matrix = malloc(nx*sizeof(matrix[0]));

	gettimeofday(&start, NULL);
	for(loop=1 ; loop<=LOOPS ; loop++)
	{
		for(i=0 ; i<nx ; i++) matrix[i] = i;
		starpu_vector_data_register(&vector_handle, 0, (uintptr_t)matrix, nx, sizeof(matrix[0]));
		ret = starpu_insert_task(vector_codelet, STARPU_RW, vector_handle, 0);
		starpu_data_unregister(vector_handle);
		if (ret == -ENODEV) return ret;
	}
	gettimeofday(&end, NULL);

	vector_timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	vector_timing /= LOOPS;
	mean = matrix[0];

	gettimeofday(&start, NULL);
	for(loop=1 ; loop<=LOOPS ; loop++)
	{
		for(i=0 ; i<nx ; i++) matrix[i] = i;
		starpu_matrix_data_register(&matrix_handle, 0, (uintptr_t)matrix, nx/2, nx/2, 2, sizeof(matrix[0]));
		ret = starpu_insert_task(matrix_codelet, STARPU_RW, matrix_handle, 0);
		starpu_data_unregister(matrix_handle);
		if (ret == -ENODEV) return ret;
	}
	gettimeofday(&end, NULL);

	matrix_timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	matrix_timing /= LOOPS;

	if (mean == matrix[0])
	{
		fprintf(stderr, "%d\t%f\t%f\n", nx, vector_timing, matrix_timing);

		{
			char *output_dir = getenv("STARPU_BENCH_DIR");
			char *bench_id = getenv("STARPU_BENCH_ID");
			if (output_dir && bench_id)
			{
				char file[1024];
				FILE *f;
				sprintf(file, "%s/matrix_as_vector_%s.dat", output_dir, device_name);
				f = fopen(file, "a");
				fprintf(f, "%s\t%d\t%f\t%f\n", bench_id, nx, vector_timing, matrix_timing);
				fclose(f);
			}
		}

		return EXIT_SUCCESS;
	}
	else
	{
		FPRINTF(stderr, "Incorrect result nx=%7d --> mean=%7f != %7f\n", nx, matrix[0], mean);
		return EXIT_FAILURE;
	}
}

#define NX_MIN 2
#define NX_MAX 1024*1024

int check_size_on_device(uint32_t where, char *device_name)
{
	int nx, ret;
	struct starpu_codelet vector_codelet;
	struct starpu_codelet matrix_codelet;

	fprintf(stderr, "# Device: %s\n", device_name);
	fprintf(stderr, "# nx vector_timing matrix_timing\n");
	starpu_codelet_init(&vector_codelet);
	vector_codelet.modes[0] = STARPU_RW;
	vector_codelet.nbuffers = 1;
	if (where == STARPU_CPU) vector_codelet.cpu_funcs[0] = vector_cpu_func;
	if (where == STARPU_CUDA) vector_codelet.cuda_funcs[0] = vector_cuda_func;
//	if (where == STARPU_OPENCL) vector_codelet.opencl_funcs[0] = vector_opencl_func;

	starpu_codelet_init(&matrix_codelet);
	matrix_codelet.modes[0] = STARPU_RW;
	matrix_codelet.nbuffers = 1;
	if (where == STARPU_CPU) matrix_codelet.cpu_funcs[0] = matrix_cpu_func;
	if (where == STARPU_CUDA) matrix_codelet.cuda_funcs[0] = matrix_cuda_func;
//	if (where == STARPU_OPENCL) matrix_codelet.opencl_funcs[0] = matrix_opencl_func;

	for(nx=NX_MIN ; nx<=NX_MAX ; nx*=2)
	{
		ret = check_size(nx, &vector_codelet, &matrix_codelet, device_name);
		if (ret != EXIT_SUCCESS) break;
	}
	return ret;

};

int main(int argc, char **argv)
{
	int ret;
	unsigned devices;

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	devices = starpu_cpu_worker_get_count();
	if (devices)
	{
		ret = check_size_on_device(STARPU_CPU, "STARPU_CPU");
		if (ret) goto error;
	}
	devices = starpu_cuda_worker_get_count();
	if (devices)
	{
		starpu_helper_cublas_init();
		ret = check_size_on_device(STARPU_CUDA, "STARPU_CUDA");
		starpu_helper_cublas_shutdown();
		if (ret) goto error;
	}
	devices = starpu_opencl_worker_get_count();
	if (devices)
	{
		ret = check_size_on_device(STARPU_OPENCL, "STARPU_OPENCL");
		if (ret) goto error;
	}

error:
	if (ret == -ENODEV) ret=STARPU_TEST_SKIPPED;
	starpu_shutdown();
	STARPU_RETURN(ret);
}
