/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <math.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_code;
void opencl_codelet(void *descr[], void *_args)
{
	(void)_args;
	cl_kernel kernel;
	cl_command_queue queue;
	int id, devid, err, n;
	cl_mem matrix = (cl_mem)STARPU_MATRIX_GET_DEV_HANDLE(descr[0]);
	cl_mem vector = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(descr[1]);
	cl_mem mult = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(descr[2]);
	int nx = STARPU_MATRIX_GET_NX(descr[0]);
	int ny = STARPU_MATRIX_GET_NY(descr[0]);
	int ld = STARPU_MATRIX_GET_LD(descr[0]);

        id = starpu_worker_get_id_check();
        devid = starpu_worker_get_devid(id);

        err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_code, "matVecMult", devid);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

        n=0;
        err = clSetKernelArg(kernel, n++, sizeof(matrix), &matrix);
        err |= clSetKernelArg(kernel, n++, sizeof(vector), &vector);
        err |= clSetKernelArg(kernel, n++, sizeof(nx), (void*)&nx);
        err |= clSetKernelArg(kernel, n++, sizeof(ny), (void*)&ny);
	err |= clSetKernelArg(kernel, n++, sizeof(mult), &mult);
	err |= clSetKernelArg(kernel, n++, sizeof(ld), (void*)&ld);
        if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t global=nx*ny;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}
	starpu_opencl_release_kernel(kernel);
}
#endif

void fillArray(float* pfData, int iSize)
{
    int i;
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i)
    {
            pfData[i] = fScale * rand();
    }
}

#if 0
void printArray(float* pfData, int iSize)
{
    int i;
    for (i = 0; i < iSize; ++i)
    {
            FPRINTF(stderr, "%f ", pfData[i]);
    }
    FPRINTF(stderr, "\n");
}
#endif

void matVecMult(const float *matrix, const float *vector, int width, int height, float *mult)
{
    int i, j;
    for (i = 0; i < height; ++i)
    {
        double sum = 0;
        for (j = 0; j < width; ++j)
	{
            double a = matrix[i * width + j];
            double b = vector[j];
            sum += a * b;
        }
        mult[i] = (float)sum;
    }
}

int compareL2fe(const float* reference, const float* data, const unsigned int len, const float epsilon)
{
    float error = 0;
    float ref = 0;
    unsigned int i;

    for(i = 0; i < len; ++i)
    {
        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) return 1;

    float normError = sqrtf(error);
    error = normError / normRef;

    return error < epsilon ? 0 : 1;
}

static struct starpu_perfmodel starpu_matvecmult_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "matvecmult"
};

static struct starpu_codelet cl =
{
#ifdef STARPU_USE_OPENCL
        .opencl_funcs[0] = opencl_codelet,
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
        .nbuffers = 3,
	.modes[0] = STARPU_R,
	.modes[1] = STARPU_R,
	.modes[2] = STARPU_RW,
	.model = &starpu_matvecmult_model
};

int main(void)
{
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	conf.ncpus = 0;
	conf.ncuda = 0;
	conf.nmic = 0;
	conf.nopencl = 1;

        /* int width=1100; */
        /* int height=244021; */
        int width=20;
        int height=4;

        float *matrix, *vector, *mult;
        float *correctResult;
        unsigned int mem_size_matrix, mem_size_vector, mem_size_mult;

	starpu_data_handle_t matrix_handle, vector_handle, mult_handle;
	int ret, submit;

        ret = starpu_init(&conf);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
                FPRINTF(stderr, "This application requires an OpenCL worker.\n");
		return 77;
	}

        mem_size_matrix = width * height * sizeof(float);
        matrix = (float*)malloc(mem_size_matrix);
        mem_size_vector = width * sizeof(float);
        vector = (float*)malloc(mem_size_vector);
        mem_size_mult = height * sizeof(float);
        mult = (float*)malloc(mem_size_mult);
        correctResult = (float*)malloc(mem_size_mult);

        assert(matrix);
        assert(vector);
        assert(mult);
        assert(correctResult);

        fillArray(matrix, width*height);
        fillArray(vector, width);
        fillArray(mult, height);
        matVecMult(matrix, vector, width, height, correctResult);

	starpu_matrix_data_register(&matrix_handle, STARPU_MAIN_RAM, (uintptr_t)matrix, width, width, height, sizeof(float));
	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, width, sizeof(float));
	starpu_vector_data_register(&mult_handle, STARPU_MAIN_RAM, (uintptr_t)mult, height, sizeof(float));

#ifdef STARPU_USE_OPENCL
        ret = starpu_opencl_load_opencl_from_file("examples/matvecmult/matvecmult_kernel.cl", &opencl_code, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

        struct starpu_task *task = starpu_task_create();
        task->cl = &cl;
        task->callback_func = NULL;
        task->handles[0] = matrix_handle;
        task->handles[1] = vector_handle;
        task->handles[2] = mult_handle;

        submit = starpu_task_submit(task);
        if (STARPU_UNLIKELY(submit == -ENODEV))
	{
                FPRINTF(stderr, "No worker may execute this task. This application requires an OpenCL worker.\n");
	}
	else
	{
		starpu_task_wait_for_all();
	}

	starpu_data_unregister(matrix_handle);
	starpu_data_unregister(vector_handle);
	starpu_data_unregister(mult_handle);

        if (STARPU_LIKELY(submit != -ENODEV))
	{
		int res = compareL2fe(correctResult, mult, height, 1e-6f);
		FPRINTF(stdout, "TEST %s\n\n", (res == 0) ? "PASSED" : "FAILED !!!");
	}

#if 0
        printArray(matrix, width*height);
        printArray(vector, width);
        printArray(mult, height);
#endif

	free(matrix);
	free(vector);
	free(mult);
	free(correctResult);
        starpu_shutdown();

	return (submit == -ENODEV) ? 77 : 0;
}
