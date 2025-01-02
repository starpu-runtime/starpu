/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "generic.h"
#include "../../../../helper.h"

struct stats global_stats;

#ifdef STARPU_USE_CPU
void cpu_func(void *buffers[], void *args)
{
	(void)buffers;
	(void)args;

	STARPU_SKIP_IF_VALGRIND;

	global_stats.cpu++;
}
#endif /* !STARPU_USE_CPU */

#ifdef STARPU_USE_CUDA
void cuda_func(void *buffers[], void *args)
{
	(void)buffers;
	(void)args;

	STARPU_SKIP_IF_VALGRIND;

	global_stats.cuda++;
}

void cpu_to_cuda_func(void *buffers[], void *args)
{
	(void)buffers;
	(void)args;

	STARPU_SKIP_IF_VALGRIND;

	global_stats.cpu_to_cuda++;
}

void cuda_to_cpu_func(void *buffers[], void *args)
{
	(void)buffers;
	(void)args;

	STARPU_SKIP_IF_VALGRIND;

	global_stats.cuda_to_cpu++;
}

struct starpu_codelet cpu_to_cuda_cl =
{
	.cuda_funcs = {cpu_to_cuda_func},
	.nbuffers = 1,
	.modes = { STARPU_RW },
};

struct starpu_codelet cuda_to_cpu_cl =
{
	.cpu_funcs = {cuda_to_cpu_func},
	.nbuffers = 1,
	.modes = { STARPU_RW },
};
#endif /* !STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
void opencl_func(void *buffers[], void *args)
{
	(void)buffers;
	(void)args;

	STARPU_SKIP_IF_VALGRIND;

	global_stats.opencl++;
}

static
void cpu_to_opencl_func(void *buffers[], void *args)
{
	(void)buffers;
	(void)args;

	STARPU_SKIP_IF_VALGRIND;

	global_stats.cpu_to_opencl++;
}

static
void opencl_to_cpu_func(void *buffers[], void *args)
{
	(void)buffers;
	(void)args;

	STARPU_SKIP_IF_VALGRIND;

	global_stats.opencl_to_cpu++;
}

struct starpu_codelet cpu_to_opencl_cl =
{
	.opencl_funcs = {cpu_to_opencl_func},
	.nbuffers = 1,
	.modes = { STARPU_RW },
};

struct starpu_codelet opencl_to_cpu_cl =
{
	.cpu_funcs = {opencl_to_cpu_func},
	.nbuffers = 1,
	.modes = { STARPU_RW },
};
#endif /* !STARPU_USE_OPENCL */

struct starpu_multiformat_data_interface_ops ops =
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

void
print_stats(struct stats *s)
{
#ifdef STARPU_USE_CPU
	FPRINTF(stderr, "cpu         : %u\n", s->cpu);
#endif /* !STARPU_USE_CPU */
#ifdef STARPU_USE_CUDA
	FPRINTF(stderr, "cuda        : %u\n"
			"cpu->cuda   : %u\n"
			"cuda->cpu   : %u\n",
			s->cuda,
			s->cpu_to_cuda,
			s->cuda_to_cpu);
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	FPRINTF(stderr, "opencl      : %u\n"
			"cpu->opencl : %u\n"
			"opencl->cpu : %u\n",
			s->opencl,
			s->cpu_to_opencl,
			s->opencl_to_cpu);
#endif /* !STARPU_USE_OPENCL */
}

void reset_stats(struct stats *s)
{
#ifdef STARPU_USE_CPU
	s->cpu = 0;
#endif /* !STARPU_USE_CPU */
#ifdef STARPU_USE_CUDA
	s->cuda = 0;
	s->cpu_to_cuda = 0;
	s->cuda_to_cpu = 0;
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	s->opencl = 0;
	s->cpu_to_opencl = 0;
	s->opencl_to_cpu = 0;
#endif /* !STARPU_USE_OPENCL */
}

int
compare_stats(struct stats *s1, struct stats *s2)
{
	if (
#ifdef STARPU_USE_CPU
	    s1->cpu == s2->cpu &&
#endif /* !STARPU_USE_CPU */
#ifdef STARPU_USE_CUDA
	    s1->cuda == s2->cuda &&
	    s1->cpu_to_cuda == s2->cpu_to_cuda &&
	    s1->cuda_to_cpu == s2->cuda_to_cpu &&
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	    s1->opencl == s2->opencl &&
	    s1->cpu_to_opencl == s2->cpu_to_opencl &&
	    s1->opencl_to_cpu == s2->opencl_to_cpu &&
#endif /* !STARPU_USE_OPENCL */
	    1 /* Just so the build does not fail if we disable EVERYTHING */
	)
		return 0;
	else
		return 1;

}
