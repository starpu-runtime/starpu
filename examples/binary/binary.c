/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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
#include <pthread.h>
#include <sys/time.h>

#define FPRINTF(ofile, fmt, args ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ##args); }} while(0)

#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
extern void opencl_codelet(void *descr[], __attribute__ ((unused)) void *_args);
struct starpu_opencl_program opencl_program;
#endif

struct starpu_codelet cl =
{
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_codelet, NULL},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

int compute(char *file_name, int load_as_file)
{
	float float_array[4] __attribute__ ((aligned (16))) = { 0.0f, 0.0f, 0.0f, 0.0f};
	starpu_data_handle_t float_array_handle;
	unsigned i;
	int ret = 0;
	unsigned niter = 500;

	starpu_vector_data_register(&float_array_handle, 0, (uintptr_t)&float_array, 4, sizeof(float));

#ifdef STARPU_USE_OPENCL
	if (load_as_file)
	{
		ret = starpu_opencl_compile_opencl_from_file(file_name, NULL);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_compile_opencl_from_file");
		ret = starpu_opencl_load_binary_opencl(file_name, &opencl_program);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_binary_opencl");
	}
	else
	{
		char located_file_name[1024];
		char located_dir_name[1024];
		char opencl_program_source[16384];
		starpu_opencl_load_program_source(file_name, located_file_name, located_dir_name, opencl_program_source);
		ret = starpu_opencl_compile_opencl_from_string(opencl_program_source, "incrementer", NULL);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_compile_opencl_from_file");
		ret = starpu_opencl_load_binary_opencl("incrementer", &opencl_program);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_binary_opencl");
	}
#endif

	for (i = 0; i < niter; i++)
	{
		ret = starpu_insert_task(&cl, STARPU_RW, float_array_handle, 0);
		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			FPRINTF(stderr, "No worker may execute this task\n");
			exit(0);
		}
	}

	starpu_task_wait_for_all();

	/* update the array in RAM */
	starpu_data_unregister(float_array_handle);

	FPRINTF(stderr, "array -> %f, %f, %f, %f\n", float_array[0], float_array[1], float_array[2], float_array[3]);

	if (float_array[0] != niter || float_array[0] != float_array[1] + float_array[2] + float_array[3])
	{
		FPRINTF(stderr, "Incorrect result\n");
		ret = 1;
	}
	return ret;
}

int main(int argc, char **argv)
{
	int ret = 0;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	conf.ncpus = 0;
	conf.ncuda = 0;

        ret = starpu_init(&conf);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
                FPRINTF(stderr, "This application requires an OpenCL worker.\n");
		starpu_shutdown();
		return 77;
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	ret = compute("examples/incrementer/incrementer_kernels_opencl_kernel.cl", 1);
	if (ret == 0)
		ret = compute("examples/incrementer/incrementer_kernels_opencl_kernel.cl", 0);

	starpu_shutdown();
	return ret;
}
