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

extern void cpu_codelet(void *descr[], void *_args);
#ifdef STARPU_USE_CUDA
extern void cuda_codelet(void *descr[], void *_args);
#endif
#ifdef STARPU_USE_OPENCL
extern void opencl_codelet(void *descr[], void *_args);
struct starpu_opencl_program opencl_code;
#endif

typedef void (*device_func)(void **, void *);

int execute_on(uint32_t where, device_func func, float *block, int pnx, int pny, int pnz, float multiplier)
{
	starpu_codelet cl;
	starpu_data_handle block_handle;
        int i, j, k;

	starpu_block_data_register(&block_handle, 0, (uintptr_t)block, pnx, pnx*pny, pnx, pny, pnz, sizeof(float));

	cl.where = where;
        cl.cuda_func = func;
        cl.cpu_func = func;
        cl.opencl_func = func;
        cl.nbuffers = 1;
        cl.model = NULL;

        struct starpu_task *task = starpu_task_create();
        task->cl = &cl;
        task->callback_func = NULL;
        task->buffers[0].handle = block_handle;
        task->buffers[0].mode = STARPU_RW;
	task->cl_arg = &multiplier;

        int ret = starpu_task_submit(task);
        if (STARPU_UNLIKELY(ret == -ENODEV)) {
                fprintf(stderr, "No worker may execute this task\n");
                return 1;
	}

	starpu_task_wait_for_all();

	/* update the array in RAM */
        starpu_data_acquire(block_handle, STARPU_R);

        for(i=0 ; i<pnx*pny*pnz; i++) {
          fprintf(stderr, "%f ", block[i]);
        }
        fprintf(stderr, "\n");

        starpu_data_release(block_handle);

        return 0;
}

int main(int argc, char **argv)
{
	starpu_codelet cl;
        float *block;
        int i, ret;
        int nx=3;
        int ny=2;
        int nz=4;
        float multiplier=1.0;

        starpu_init(NULL);

        block = (float*)malloc(nx*ny*nz*sizeof(float));
        assert(block);
        for(i=0 ; i<nx*ny*nz ; i++) block[i] = i+1;

        ret = execute_on(STARPU_CPU, cpu_codelet, block, nx, ny, nz, 1.0);
        if (!ret) multiplier *= 1.0;
#ifdef STARPU_USE_OPENCL
        starpu_opencl_load_opencl_from_file("examples/basic_examples/block_opencl_kernel.cl", &opencl_code);
        ret = execute_on(STARPU_OPENCL, opencl_codelet, block, nx, ny, nz, 2.0);
        if (!ret) multiplier *= 2.0;
#endif
#ifdef STARPU_USE_CUDA
        ret = execute_on(STARPU_CUDA, cuda_codelet, block, nx, ny, nz, 3.0);
        if (!ret) multiplier *= 3.0;
#endif

        // Check result is correct
        ret=1;
        for(i=0 ; i<nx*ny*nz ; i++) {
          if (block[i] != (i+1) * multiplier) {
            ret=0;
            break;
          }
        }

        fprintf(stderr,"TEST %s\n", ret==1?"PASSED":"FAILED");
        starpu_shutdown();

	return 0;
}
