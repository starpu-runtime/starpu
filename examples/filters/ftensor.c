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

void cpu_func(void *buffers[], void *cl_arg)
{
    int i, j, k, l;
    int *factor = (int *) cl_arg;
    int *val = (int *)STARPU_TENSOR_GET_PTR(buffers[0]);
    int nx = (int)STARPU_TENSOR_GET_NX(buffers[0]);
    int ny = (int)STARPU_TENSOR_GET_NY(buffers[0]);
    int nz = (int)STARPU_TENSOR_GET_NZ(buffers[0]);
    int nt = (int)STARPU_TENSOR_GET_NT(buffers[0]);
    unsigned ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
    unsigned ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
    unsigned ldt = STARPU_TENSOR_GET_LDT(buffers[0]);

    for(l=0; l<nt ; l++)
    {
        for(k=0; k<nz ; k++)
        {
            for(j=0; j<ny ; j++)
            {
                for(i=0; i<nx ; i++)
                    val[(l*ldt)+(k*ldz)+(j*ldy)+i] = *factor;
            }
        }
    }
        
}

void print_tensor(int *tensor, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt)
{
        int i, j, k, l;
        FPRINTF(stderr, "tensor=%p nx=%d ny=%d nz=%d nt=%d ldy=%u ldz=%u ldt=%u\n", tensor, nx, ny, nz, nt, ldy, ldz, ldt);
        for(l=0 ; l<nt ; l++)
        {
            for(k=0 ; k<nz ; k++)
            {
                for(j=0 ; j<ny ; j++)
                {
                    for(i=0 ; i<nx ; i++)
                    {
                        FPRINTF(stderr, "%2d ", tensor[(l*ldt)+(k*ldz)+(j*ldy)+i]);
                    }
                    FPRINTF(stderr,"\n");
                }
                FPRINTF(stderr,"\n");
            }
            FPRINTF(stderr,"\n");
        }
        FPRINTF(stderr,"\n");
}

void print_data(starpu_data_handle_t tensor_handle)
{
    int *tensor = (int *)starpu_tensor_get_local_ptr(tensor_handle);
    int nx = starpu_tensor_get_nx(tensor_handle);
    int ny = starpu_tensor_get_ny(tensor_handle);
    int nz = starpu_tensor_get_nz(tensor_handle);
    int nt = starpu_tensor_get_nt(tensor_handle);
    unsigned ldy = starpu_tensor_get_local_ldy(tensor_handle);
    unsigned ldz = starpu_tensor_get_local_ldz(tensor_handle);
    unsigned ldt = starpu_tensor_get_local_ldt(tensor_handle);

    print_tensor(tensor, nx, ny, nz, nt, ldy, ldz, ldt);
}

int main(void)
{
    int *tensor,n=0;
    int i, j, k, l;
    int ret;

    tensor = (int*)malloc(NX*NY*NZ*NT*sizeof(tensor[0]));
    assert(tensor);
    for(l=0 ; l<NT ; l++)
    {
        for(k=0 ; k<NZ ; k++)
        {
            for(j=0 ; j<NY ; j++)
            {
                for(i=0 ; i<NX ; i++)
                {
                    tensor[(l*NX*NY*NZ)+(k*NX*NY)+(j*NX)+i] = n++;
                }
            }
        }
    }

    starpu_data_handle_t handle;
    struct starpu_codelet cl =
    {
        .cpu_funcs = {cpu_func},
        .cpu_funcs_name = {"cpu_func"},
        .nbuffers = 1,
        .modes = {STARPU_RW},
        .name = "tensor_scal"
    };

    ret = starpu_init(NULL);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
    
    /* Declare data to StarPU */
    starpu_tensor_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)tensor, NX, NX*NY, NX*NY*NZ, NX, NY, NZ, NT, sizeof(int));
    FPRINTF(stderr, "IN  Tensor\n");
    print_data(handle);

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
        print_data(stensor);
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
        if (ret)
        {
            FPRINTF(stderr, "Error when submitting task\n");
            exit(ret);
        }
    }

    /* Unpartition the data, unregister it from StarPU and shutdown */
    starpu_data_unpartition(handle, STARPU_MAIN_RAM);
    print_data(handle);
    starpu_data_unregister(handle);

    /* Print result tensor */
    FPRINTF(stderr, "OUT Tensor\n");
    print_tensor(tensor, NX, NY, NZ, NT, NX, NX*NY, NX*NY*NZ);

    free(tensor);

    starpu_shutdown();
    return 0;

}    