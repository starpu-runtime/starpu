/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#ifndef TEST_INTERFACES_H
#define TEST_INTERFACES_H

#include "../../helper.h"

/*
 * Users do not know about this enum. They only know that SUCCESS is 0, and
 * FAILURE is 1. Therefore, the values of SUCCESS and FAILURE shall not be
 * changed.
 */
enum exit_code
{
	SUCCESS                 = 0,
	FAILURE                 = 1,
	UNTESTED                = 2,
	NO_DEVICE		= 3,
	TASK_SUBMISSION_FAILURE = 4
};

struct test_config
{
	/** we use pointers as we want to allow static initializations in the main application */
	/* A pointer to a registered handle */
	starpu_data_handle_t *handle;

	/* A pointer to a registered handle, that will be used to test
	 * RAM to RAM copy. The values it points to should be different from
	 * the ones pointed to by the previous handle. */
	starpu_data_handle_t *dummy_handle;

	/* StarPU codelets. The following functions should :
	 * 1) Check that the values are correct
	 * 2) Negate every element
	 */
	starpu_cpu_func_t cpu_func;
	starpu_cuda_func_t cuda_func;
	starpu_opencl_func_t opencl_func;
	char *cpu_func_name;

	/* The previous codelets must update this field at the end of their
	 * execution. copy_failed must be FAILURE if the copy failed, SUCCESS otherwise. */
	enum exit_code copy_failed;

	/* A human-readable name for the test */
	const char *name;
};

struct data_interface_test_summary
{
	int success;

	/* Copy methods */
	int cpu_to_cpu;
	int cpu_to_cuda;
	int cuda_to_cuda;
	int cuda_to_cpu;
	int cpu_to_cuda_async;
	int cuda_to_cpu_async;
	int cuda_to_cuda_async;
	int cpu_to_opencl;
	int opencl_to_cpu;
	int cpu_to_opencl_async;
	int opencl_to_cpu_async;
	int cpu_to_mic;
	int mic_to_cpu;
	int cpu_to_mic_async;
	int mic_to_cpu_async;

	/* Other stuff */
	int compare;
	int to_pointer;
	int pointer_is_inside;
	int pack;
};

void data_interface_test_summary_print(FILE *f, struct data_interface_test_summary *summary);
int data_interface_test_summary_success(struct data_interface_test_summary *summary);

void run_tests(struct test_config*, struct data_interface_test_summary *summary);

#endif /* !TEST_INTERFACES_H */
