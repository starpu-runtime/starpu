/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../test_interfaces.h"
#include "../../../helper.h"

#define NX 4
#define NY NX
#define NZ NX
#define NT NX

/* Prototypes */
static void register_data(void);
static void unregister_data(void);
void test_tensor_cpu_func(void *buffers[], void *args);
#ifdef STARPU_USE_CUDA
extern void test_tensor_cuda_func(void *buffers[], void *_args);
#endif
#ifdef STARPU_USE_OPENCL
extern void test_tensor_opencl_func(void *buffers[], void *args);
#endif

static starpu_data_handle_t _tensor_handle;
static starpu_data_handle_t _tensor2_handle;

static int _tensor[NX*NY*NZ*NT];
static int _tensor2[NX*NY*NZ*NT];

struct test_config tensor_config =
{
	.cpu_func      = test_tensor_cpu_func,
#ifdef STARPU_USE_CUDA
	.cuda_func     = test_tensor_cuda_func,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func   = test_tensor_opencl_func,
#endif
	.handle        = &_tensor_handle,
	.ptr           = _tensor,
	.dummy_handle  = &_tensor2_handle,
	.dummy_ptr     = _tensor2,
	.copy_failed   = SUCCESS,
	.name          = "tensor_interface"
};

static void
register_data(void)
{
	/* Initializing data */
	int val = 0;
	int i, j, k, l;
	for (l = 0; l < NT; l++)
	    for (k = 0; k < NZ; k++)
		for (j = 0; j < NY; j++)
			for (i = 0; i < NX; i++)
				_tensor[(l*NX*NY*NZ)+(k*NX*NY)+(j*NX)+i] = val++;

	/* Registering data */
	starpu_tensor_data_register(&_tensor_handle,
                                    STARPU_MAIN_RAM,
                                    (uintptr_t)_tensor,
				    NX,
				    NX * NY,
				    NX * NY * NZ,
				    NX,
				    NY,
				    NZ,
				    NT,
				    sizeof(_tensor[0]));
	starpu_tensor_data_register(&_tensor2_handle,
                                    STARPU_MAIN_RAM,
                                    (uintptr_t)_tensor2,
				    NX,
				    NX * NY,
				    NX * NY * NZ,
				    NX,
				    NY,
				    NZ,
				    NT,
				    sizeof(_tensor2[0]));
}

static void
unregister_data(void)
{
	starpu_data_unregister(_tensor_handle);
	starpu_data_unregister(_tensor2_handle);
}

void test_tensor_cpu_func(void *buffers[], void *args)
{
	STARPU_SKIP_IF_VALGRIND;

	int factor = *(int*)args;
	int nx = STARPU_TENSOR_GET_NX(buffers[0]);
	int ny = STARPU_TENSOR_GET_NY(buffers[0]);
	int nz = STARPU_TENSOR_GET_NZ(buffers[0]);
	int nt = STARPU_TENSOR_GET_NT(buffers[0]);
        unsigned ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
        unsigned ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
        unsigned ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
	int *tensor = (int *) STARPU_TENSOR_GET_PTR(buffers[0]);
	int i, j, k, l;
	int val = 0;
	tensor_config.copy_failed = SUCCESS;
	for (l = 0; l < nt; l++)
	{
	    for (k = 0; k < nz; k++)
	    {
		for (j = 0; j < ny; j++)
		{
			for (i = 0; i < nx; i++)
			{
				if (tensor[(l*ldt)+(k*ldz)+(j*ldy)+i] != factor * val)
				{
					tensor_config.copy_failed = FAILURE;
					return;
				}
				else
				{
					tensor[(l*ldt)+(k*ldz)+(j*ldy)+i] *= -1;
					val++;
				}
			}
		}
	    }
	}
}

int
main(int argc, char **argv)
{
	struct data_interface_test_summary summary;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.ncuda = 2;
	conf.nopencl = 1;

	int ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	register_data();

	ret = run_tests(&tensor_config, &summary);

	unregister_data();

	starpu_shutdown();

	if (ret) data_interface_test_summary_print(stderr, &summary);

	return data_interface_test_summary_success(&summary);
}

