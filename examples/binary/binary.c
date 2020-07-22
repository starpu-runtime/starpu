/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This shows how to load OpenCL programs, either from a .cl file, or from a
 * string containing the program itself.
 */

#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#ifdef STARPU_USE_OPENCL
extern void opencl_codelet(void *descr[], void *_args);
struct starpu_opencl_program opencl_program;
#endif

struct starpu_codelet cl =
{
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {opencl_codelet},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

int compute(char *file_name, int load_as_file, int with_malloc)
{
	float float_array[4] STARPU_ATTRIBUTE_ALIGNED(16) = { 0.0f, 0.0f, 0.0f, 0.0f};
	starpu_data_handle_t float_array_handle;
	unsigned i;
	int ret = 0;
#ifdef STARPU_QUICK_CHECK
	unsigned niter = 50;
#else
	unsigned niter = 500;
#endif

	starpu_vector_data_register(&float_array_handle, STARPU_MAIN_RAM, (uintptr_t)&float_array, 4, sizeof(float));

#ifdef STARPU_USE_OPENCL
	if (load_as_file)
	{
		ret = starpu_opencl_compile_opencl_from_file(file_name, NULL);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_compile_opencl_from_file");
		ret = starpu_opencl_load_binary_opencl(file_name, &opencl_program);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_binary_opencl");
	}
	else if (with_malloc)
	{
		char *located_file_name;
		char *located_dir_name;
		char *opencl_program_source;
		starpu_opencl_load_program_source_malloc(file_name, &located_file_name, &located_dir_name, &opencl_program_source);
		ret = starpu_opencl_compile_opencl_from_string(opencl_program_source, "incrementer", NULL);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_compile_opencl_from_file");
		ret = starpu_opencl_load_binary_opencl("incrementer", &opencl_program);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_binary_opencl");
		free(located_file_name);
		free(located_dir_name);
		free(opencl_program_source);
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
		ret = starpu_task_insert(&cl, STARPU_RW, float_array_handle, STARPU_TAG_ONLY, (starpu_tag_t) i, 0);
		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			FPRINTF(stderr, "No worker may execute this task\n");
			exit(0);
		}
	}

	starpu_task_wait_for_all();

	/* update the array in RAM */
	starpu_data_unregister(float_array_handle);

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_unload_opencl(&opencl_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif

	FPRINTF(stderr, "array -> %f, %f, %f, %f\n", float_array[0], float_array[1], float_array[2], float_array[3]);

	if (float_array[0] != niter || float_array[0] != float_array[1] + float_array[2] + float_array[3])
	{
		FPRINTF(stderr, "Incorrect result\n");
		ret = 1;
	}
	return ret;
}

int main(void)
{
	int ret = 0;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	conf.ncuda = 0;

        ret = starpu_init(&conf);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
                FPRINTF(stderr, "This application requires an OpenCL worker.\n");
		return 77;
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	if (starpu_opencl_worker_get_count() == 0)
	{
                FPRINTF(stderr, "This application requires an OpenCL worker.\n");
		starpu_shutdown();
		return 77;
	}

	ret = compute("examples/incrementer/incrementer_kernels_opencl_kernel.cl", 1, -1);
	if (ret == 0)
		ret = compute("examples/incrementer/incrementer_kernels_opencl_kernel.cl", 0, 0);
	else
		FPRINTF(stderr, "Error when calling compute %d\n", ret);
	if (ret == 0)
	     ret = compute("examples/incrementer/incrementer_kernels_opencl_kernel.cl", 0, 1);
	else
		FPRINTF(stderr, "Error when calling compute %d\n", ret);

	starpu_shutdown();
	return ret;
}
