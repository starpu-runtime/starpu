/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); fflush(ofile); }} while(0)

void cpu_codeletA(void *descr[], void *args);
void cpu_codeletB(void *descr[], void *args);

struct starpu_codelet clA =
{
	.cpu_funcs = {cpu_codeletA},
	.cpu_funcs_name = {"cpu_codeletA"},
	.name = "codeletA"
};

struct starpu_codelet clB =
{
	.cpu_funcs = {cpu_codeletB},
	.cpu_funcs_name = {"cpu_codeletB"},
	.name = "codeletB"
};

void cpu_codeletA(void *descr[], void *args)
{
	(void)descr;
	starpu_tag_t tagB;

	FPRINTF(stderr, "[Task A] before releasing B\n");

	starpu_codelet_unpack_args(args, &tagB);
	starpu_tag_notify_from_apps(tagB);

	FPRINTF(stderr, "[Task A] after releasing B\n");
}

void cpu_codeletB(void *descr[], void *args)
{
	(void)descr;
	(void)args;

	FPRINTF(stderr, "[Task B]\n");
}

int main(void)
{
	int ret;
	starpu_tag_t tagA = 421;
	starpu_tag_t tagB = 842;

	struct starpu_conf conf;

	if (sizeof(starpu_tag_t) > sizeof(void*))
	{
		// Can't pass a tag_t through callback arg :/
		return 77;
	}

	starpu_conf_init(&conf);
	conf.nmpi_sc = 0;
	conf.ntcpip_sc = 0;

	ret = starpu_init(&conf);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		return 77;
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() < 1)
	{
		FPRINTF(stderr, "This application requires at least 1 cpu worker\n");
		starpu_shutdown();
		return 77;
	}

	starpu_tag_declare_deps_array(tagB, 1, &tagA);

	ret = starpu_task_insert(&clA,
				 STARPU_TAG, tagA,
				 STARPU_VALUE, &tagB, sizeof(starpu_tag_t),
				 STARPU_NAME, "taskA",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	ret = starpu_task_insert(&clB,
				 STARPU_TAG, tagB,
				 STARPU_NAME, "taskB",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	// Release taskA
	//starpu_tag_notify_from_apps(tagA);

	starpu_shutdown();

	return ret;
}
