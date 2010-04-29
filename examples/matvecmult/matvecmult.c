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

#include <starpu.h>
#include <starpu_opencl.h>
#include <pthread.h>
#include <math.h>

//static int width=1100;
//static int height=244021;
static int width=20;
static int height=4;

#ifdef STARPU_USE_OPENCL
void opencl_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	cl_kernel kernel;
	cl_command_queue queue;
	int id, devid, err, n;
	float *matrix = (float *)STARPU_GET_MATRIX_PTR(descr[0]);
	float *vector = (float *)STARPU_GET_VECTOR_PTR(descr[1]);
	float *mult = (float *)STARPU_GET_VECTOR_PTR(descr[2]);

        id = starpu_get_worker_id();
        devid = starpu_get_worker_devid(id);

        err = starpu_opencl_load_kernel(&kernel, &queue,
                                        "examples/matvecmult/matvecmult_kernel.cl", "matVecMult", devid);
        if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = 0;
        n=0;
        err = clSetKernelArg(kernel, n++, sizeof(cl_mem), &matrix);
        err |= clSetKernelArg(kernel, n++, sizeof(cl_mem), &vector);
        err |= clSetKernelArg(kernel, n++, sizeof(int), (void*)&width);
        err |= clSetKernelArg(kernel, n++, sizeof(int), (void*)&height);
        err |= clSetKernelArg(kernel, n++, sizeof(cl_mem), &mult);
        if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
                size_t global=512;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	clFinish(queue);

	starpu_opencl_release(kernel);
}

void fillArray(float* pfData, int iSize) {
    int i;
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i) {
            pfData[i] = fScale * rand();
    }
}

void printArray(float* pfData, int iSize) {
    int i;
    for (i = 0; i < iSize; ++i) {
            fprintf(stderr, "%f ", pfData[i]);
    }
    fprintf(stderr, "\n");
}

void matVecMult(const float *matrix, const float *vector, int width, int height, float *mult) {
    int i, j;
    for (i = 0; i < height; ++i) {
        double sum = 0;
        for (j = 0; j < width; ++j) {
            double a = matrix[i * width + j];
            double b = vector[j];
            sum += a * b;
        }
        mult[i] = (float)sum;
    }
}

int compareL2fe(const float* reference, const float* data, const unsigned int len, const float epsilon) {
    float error = 0;
    float ref = 0;
    unsigned int i;

    for(i = 0; i < len; ++i) {
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
#endif

int main(int argc, char **argv)
{
	starpu_codelet cl;
        starpu_init(NULL);

#ifdef STARPU_USE_OPENCL
        float *matrix, *vector, *mult;
        float *correctResult;
        unsigned int mem_size_matrix, mem_size_vector, mem_size_mult;

	starpu_data_handle matrix_handle, vector_handle, mult_handle;

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

	starpu_register_matrix_data(&matrix_handle, 0, (uintptr_t)matrix, width, width, height, sizeof(float));
	starpu_register_vector_data(&vector_handle, 0, (uintptr_t)vector, width, sizeof(float));
	starpu_register_vector_data(&mult_handle, 0, (uintptr_t)mult, height, sizeof(float));

        _starpu_opencl_compile_source_to_opencl("examples/matvecmult/matvecmult_kernel.cl");

	cl.where = STARPU_OPENCL;
        cl.opencl_func = opencl_codelet;
        cl.nbuffers = 3;
        cl.model = NULL;

        struct starpu_task *task = starpu_task_create();
        task->cl = &cl;
        task->callback_func = NULL;
        task->buffers[0].handle = matrix_handle;
        task->buffers[0].mode = STARPU_R;
        task->buffers[1].handle = vector_handle;
        task->buffers[1].mode = STARPU_R;
        task->buffers[2].handle = mult_handle;
        task->buffers[2].mode = STARPU_RW;

        int ret = starpu_submit_task(task);
        if (STARPU_UNLIKELY(ret == -ENODEV)) {
                fprintf(stderr, "No worker may execute this task\n");
                exit(0);
	}

	starpu_wait_all_tasks();

	/* update the array in RAM */
        starpu_sync_data_with_mem(matrix_handle, STARPU_R);
        starpu_sync_data_with_mem(vector_handle, STARPU_R);
        starpu_sync_data_with_mem(mult_handle, STARPU_R);

        int res = compareL2fe(correctResult, mult, height, 1e-6f);
        printf("TEST %s\n\n", (res == 0) ? "PASSED" : "FAILED !!!");
#if 0
        printArray(matrix, width*height);
        printArray(vector, width);
        printArray(mult, height);
#endif
        starpu_release_data_from_mem(matrix_handle);
        starpu_release_data_from_mem(vector_handle);
        starpu_release_data_from_mem(mult_handle);

        starpu_shutdown();
#endif

	return 0;
}
