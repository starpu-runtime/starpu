/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This examplifies how to use partitioning filters.  We here just split a 4D
 * matrix into 4D slices (along the X axis), and run a dumb kernel on them.
 */

#include <starpu.h>

#define NX    6
#define NY    5
#define NZ    4
#define NT    3
#define PARTS 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void tensor_cpu_func(void *buffers[], void *cl_arg);

#ifdef STARPU_USE_CUDA
extern void tensor_cuda_func(void *buffers[], void *cl_arg);
#endif
#ifdef STARPU_USE_HIP
extern void tensor_hip_func(void *buffers[], void *cl_arg);
#endif


extern void generate_tensor_data(int *tensor, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt);
extern void print_tensor(int *tensor, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt);
extern void print_tensor_data(starpu_data_handle_t tensor_handle);

int main(void)
{
    int *tensor;
    int i, j, k, l;
    int ret;

    starpu_data_handle_t handle;
    struct starpu_codelet cl =
    {
        .cpu_funcs = {tensor_cpu_func},
        .cpu_funcs_name = {"tensor_cpu_func"},
#ifdef STARPU_USE_CUDA
        .cuda_funcs = {tensor_cuda_func},
        .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
        .hip_funcs = {tensor_hip_func},
        .hip_flags = {STARPU_HIP_ASYNC},
#endif
        .nbuffers = 1,
        .modes = {STARPU_RW},
        .name = "tensor_scal"
    };

    ret = starpu_init(NULL);
    if (ret == -ENODEV)
	exit(77);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_malloc((void **)&tensor, NX*NY*NZ*NT*sizeof(int));
    assert(tensor);
    generate_tensor_data(tensor, NX, NY, NZ, NT, NX, NX*NY, NX*NY*NZ);
    
    /* Declare data to StarPU */
    starpu_tensor_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)tensor, NX, NX*NY, NX*NY*NZ, NX, NY, NZ, NT, sizeof(int));
    FPRINTF(stderr, "IN  Tensor\n");
    print_tensor_data(handle);

    /* Partition the tensor in PARTS sub-tensors */
    struct starpu_data_filter f =
    {
        .filter_func = starpu_tensor_filter_block,
        .nchildren = PARTS
    };
    starpu_data_partition(handle, &f);

    FPRINTF(stderr,"Nb of partitions : %d\n",starpu_data_get_nb_children(handle));

    for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
    {
        starpu_data_handle_t stensor = starpu_data_get_sub_data(handle, 1, i);
        FPRINTF(stderr, "Sub tensor %d\n", i);
        print_tensor_data(stensor);
    }

    /* Submit a task on each sub-tensor */
    for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
    {
        int multiplier=i;
        struct starpu_task *task = starpu_task_create();

        FPRINTF(stderr,"Dealing with sub-tensor %d\n", i);
        task->cl = &cl;
        task->synchronous = 1;
        task->callback_func = NULL;
        task->handles[0] = starpu_data_get_sub_data(handle, 1, i);
        task->cl_arg = &multiplier;
        task->cl_arg_size = sizeof(multiplier);

        ret = starpu_task_submit(task);
        if (ret == -ENODEV) goto enodev;
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }

    /* Unpartition the data, unregister it from StarPU and shutdown */
    starpu_data_unpartition(handle, STARPU_MAIN_RAM);
    print_tensor_data(handle);
    starpu_data_unregister(handle);

    /* Print result tensor */
    FPRINTF(stderr, "OUT Tensor\n");
    print_tensor(tensor, NX, NY, NZ, NT, NX, NX*NY, NX*NY*NZ);

    starpu_free_noflag(tensor, NX*NY*NZ*NT*sizeof(int));

    starpu_shutdown();
    return 0;

enodev:
    FPRINTF(stderr, "WARNING: No one can execute this task\n");
    starpu_shutdown();
    return 77;
}
