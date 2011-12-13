/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  INRIA
 * Copyright (C) 2011  Centre National de la Recherche Scientifique
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
#include "../helper.h"

#define NX 16

static int vector[NX];
static starpu_data_handle_t handle;

#define ENTER() do { FPRINTF(stderr, "Entering %s\n", __func__); } while (0)

/* Counting the calls to the codelets */
struct stats
{
	unsigned int cpu;
#ifdef STARPU_USE_CUDA
	unsigned int cuda;
	unsigned int cpu_to_cuda;
	unsigned int cuda_to_cpu;
#endif
#ifdef STARPU_USE_OPENCL
	unsigned int opencl;
	unsigned int cpu_to_opencl;
	unsigned int opencl_to_cpu;
#endif
};

static struct stats global_stats;

/* "Fake" conversion codelets */
#ifdef STARPU_USE_CUDA
static void cpu_to_cuda_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cpu_to_cuda++;
}

static void cuda_to_cpu_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cuda_to_cpu++;
}

static struct starpu_codelet cpu_to_cuda_cl =
{
	.where = STARPU_CUDA,
	.cuda_funcs = {cpu_to_cuda_func, NULL},
	.nbuffers = 1
};

static struct starpu_codelet cuda_to_cpu_cl =
{
	.where = STARPU_CPU,
	.cpu_funcs = {cuda_to_cpu_func, NULL},
	.nbuffers = 1
};
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static void cpu_to_opencl_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cpu_to_opencl++;
}

static void opencl_to_cpu_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.opencl_to_cpu++;
}

static struct starpu_codelet cpu_to_opencl_cl =
{
	.where = STARPU_OPENCL,
	.opencl_funcs = {cpu_to_opencl_func, NULL},
	.nbuffers = 1
};

static struct starpu_codelet opencl_to_cpu_cl =
{
	.where = STARPU_CPU,
	.cpu_funcs = {opencl_to_cpu_func, NULL},
	.nbuffers = 1
};
#endif /* !STARPU_USE_OPENCL */

static struct starpu_multiformat_data_interface_ops ops =
{
#ifdef STARPU_USE_CUDA
	.cuda_elemsize = sizeof(int),
	.cpu_to_cuda_cl = &cpu_to_cuda_cl,
	.cuda_to_cpu_cl = &cuda_to_cpu_cl,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_elemsize = sizeof(int),
	.cpu_to_opencl_cl = &cpu_to_opencl_cl,
	.opencl_to_cpu_cl = &opencl_to_cpu_cl,
#endif
	.cpu_elemsize = sizeof(int)
};


static void
register_handle(void)
{
	int i;
	for (i = 0; i < NX; i++)
		vector[i] = i;
	starpu_multiformat_data_register(&handle, 0, vector, NX, &ops);
}

static void
unregister_handle(void)
{
	starpu_data_unregister(handle);
}

#ifdef STARPU_USE_CUDA
static void cuda_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.cuda++;
}
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static void opencl_func(void *buffers[], void *args)
{
	ENTER();
	global_stats.opencl++;
}
#endif /* !STARPU_USE_OPENCL */

static void
create_and_submit(int where)
{
	static struct starpu_codelet cl =
	{
#ifdef STARPU_USE_CUDA
		.cuda_funcs   = {cuda_func, NULL},
#endif
#if STARPU_USE_OPENCL
		.opencl_funcs = {opencl_func, NULL},
#endif
		.nbuffers    = 1
	};
	cl.where = where;

	struct starpu_task *task = starpu_task_create();
	task->cl = &cl;
	task->buffers[0].handle = handle;
	task->buffers[0].mode = STARPU_RW;

	/* We need to be sure the data has been copied to the GPU at the end 
	 * of this function */
	task->synchronous = 1;
	starpu_task_submit(task);
}

static void
print_stats(struct stats *s)
{
	FPRINTF(stderr, "cpu         : %d\n", s->cpu);
#ifdef STARPU_USE_CUDA
	FPRINTF(stderr, "cuda        : %d\n" 
			"cpu->cuda   : %d\n"
			"cuda->cpu   : %d\n",
			s->cuda,
			s->cpu_to_cuda,
			s->cuda_to_cpu);
#endif
#ifdef STARPU_USE_OPENCL
	FPRINTF(stderr, "opencl      : %d\n" 
			"cpu->opencl : %d\n"
			"opencl->cpu : %d\n",
			s->opencl,
			s->cpu_to_opencl,
			s->opencl_to_cpu);
#endif
}

static int
compare(struct stats *s1, struct stats *s2)
{
	if (
#ifdef STARPU_USE_CPU
	    s1->cpu == s2->cpu &&
#endif
#ifdef STARPU_USE_CUDA
	    s1->cuda == s2->cuda &&
	    s1->cpu_to_cuda == s2->cpu_to_cuda &&
	    s1->cuda_to_cpu == s2->cuda_to_cpu &&
#endif
#ifdef STARPU_USE_OPENCL
	    s1->opencl == s2->opencl &&
	    s1->cpu_to_opencl == s2->cpu_to_opencl &&
	    s1->opencl_to_cpu == s2->opencl_to_cpu &&
#endif
	    1 /* Just so the build does not fail if we disable EVERYTHING */
	)
		return 0;
	else
		return 1;

}

static int
test(void)
{
	struct stats expected_stats;
	memset(&expected_stats, 0, sizeof(expected_stats));

#ifdef STARPU_USE_CUDA
	create_and_submit(STARPU_CUDA);
	starpu_data_acquire(handle, STARPU_RW);

	expected_stats.cuda = 1;
	expected_stats.cpu_to_cuda = 1;
	expected_stats.cuda_to_cpu = 1;

	starpu_data_release(handle);
	if (compare(&global_stats, &expected_stats) != 0)
	{
		FPRINTF(stderr, "CUDA failed\n");
		print_stats(&global_stats);
		FPRINTF(stderr ,"\n");
		print_stats(&expected_stats);
		return -ENODEV;
	}
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
	create_and_submit(STARPU_OPENCL);
	starpu_data_acquire(handle, STARPU_RW);
	expected_stats.opencl = 1;
	expected_stats.cpu_to_opencl = 1;
	expected_stats.opencl_to_cpu = 1;

	starpu_data_release(handle);
	if (compare(&global_stats, &expected_stats) != 0)
	{
		FPRINTF(stderr, "OPENCL failed\n");
		print_stats(&global_stats);
		FPRINTF(stderr ,"\n");
		print_stats(&expected_stats);
		return -ENODEV;
	}

#endif /* !STARPU_USE_OPENCL */

	return 0;
}

int
main(void)
{
#ifdef STARPU_USE_CPU
	int ret;
	struct starpu_conf conf =
	{
		.ncpus = -1,
		.ncuda = 1,
		.nopencl = 1
	};
	memset(&global_stats, 0, sizeof(global_stats));
	ret = starpu_init(&conf);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	register_handle();

	int err = test();

	unregister_handle();
	starpu_shutdown();

	switch (err)
	{
		case -ENODEV:
			return STARPU_TEST_SKIPPED;
		case 0:
			return EXIT_SUCCESS;
		default:
			return EXIT_FAILURE;
	}
#else /* ! STARPU_USE_CPU */
	/* Without the CPU, there is no point in using the multiformat
	 * interface, so this test is pointless. */
	return STARPU_TEST_SKIPPED;
#endif
}
