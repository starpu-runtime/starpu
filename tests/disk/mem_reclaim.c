/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Corentin Salingue
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

#include <fcntl.h>
#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include "../helper.h"

/*
 * Try to write into disk memory
 * Use mechanism to push datas from main ram to disk ram
 * Here we stress the memory with more tasks than what the RAM can fit.
 */

#ifdef STARPU_HAVE_MEMCHECK_H
#include <valgrind/memcheck.h>
#else
#define VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(addr, size) (void)0
#endif

#ifdef STARPU_QUICK_CHECK
#  define NDATA 4
#  define NITER 16
#elif !defined(STARPU_LONG_CHECK)
#  define NDATA 32
#  define NITER 256
#else
#  define NDATA 128
#  define NITER 1024
#endif
#  define MEMSIZE 1
#  define MEMSIZE_STR "1"

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

static int (*any_to_any)(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, void *async_data);

/* We need a ram-to-ram copy for NUMA machine, use any_to_any for that */
static int ram_to_ram(void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node)
{
	return any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
}

const struct starpu_data_copy_methods my_vector_copy_data_methods_s =
{
	.ram_to_ram = ram_to_ram
};
struct starpu_data_interface_ops starpu_interface_my_vector_ops;

void starpu_my_vector_data_register(starpu_data_handle_t *handleptr, int home_node,
                        uintptr_t ptr, uint32_t nx, size_t elemsize)
{
	struct starpu_vector_interface vector =
	{
		.id = STARPU_VECTOR_INTERFACE_ID,
		.ptr = ptr,
		.nx = nx,
		.elemsize = elemsize,
                .dev_handle = ptr,
		.slice_base = 0,
                .offset = 0,
		.allocsize = nx * elemsize,
	};

	starpu_data_register(handleptr, home_node, &vector, &starpu_interface_my_vector_ops);
}

static unsigned values[NDATA];

static void zero(void *buffers[], void *args)
{
	(void)args;
	struct starpu_vector_interface *vector = (struct starpu_vector_interface *) buffers[0];
	unsigned *val = (unsigned*) STARPU_VECTOR_GET_PTR(vector);
	*val = 0;
	VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(val, STARPU_VECTOR_GET_NX(vector) * STARPU_VECTOR_GET_ELEMSIZE(vector));
}

static void inc(void *buffers[], void *args)
{
	struct starpu_vector_interface *vector = (struct starpu_vector_interface *) buffers[0];
	unsigned *val = (unsigned*) STARPU_VECTOR_GET_PTR(vector);
	unsigned i;
	starpu_codelet_unpack_args(args, &i);
	(*val)++;
	STARPU_ATOMIC_ADD(&values[i], 1);
}

static void check(void *buffers[], void *args)
{
	struct starpu_vector_interface *vector = (struct starpu_vector_interface *) buffers[0];
	unsigned *val = (unsigned*) STARPU_VECTOR_GET_PTR(vector);
	unsigned i;
	starpu_codelet_unpack_args(args, &i);
	STARPU_ASSERT_MSG(*val == values[i], "Incorrect value. Value %u should be %u (index %u)", *val, values[i], i);
}

static struct starpu_codelet zero_cl =
{
	.cpu_funcs = { zero },
	.nbuffers = 1,
	.modes = { STARPU_W },
};

static struct starpu_codelet inc_cl =
{
	.cpu_funcs = { inc },
	.nbuffers = 1,
	.modes = { STARPU_RW },
};

static struct starpu_codelet check_cl =
{
	.cpu_funcs = { check },
	.nbuffers = 1,
	.modes = { STARPU_R },
};

int dotest(struct starpu_disk_ops *ops, char *base, void (*vector_data_register)(starpu_data_handle_t *handleptr, int home_node, uintptr_t ptr, uint32_t nx, size_t elemsize), const char *text)
{
	int *A, *C;
	starpu_data_handle_t handles[NDATA];

	if (starpu_get_env_number_default("STARPU_DIDUSE_BARRIER", 0))
		/* This would hang */
		return STARPU_TEST_SKIPPED;

	FPRINTF(stderr, "Testing <%s>\n", text);
	/* Initialize StarPU without GPU devices to make sure the memory of the GPU devices will not be used */
	// Ignore environment variables as we want to force the exact number of workers
	struct starpu_conf conf;
	int ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return EXIT_FAILURE;
	conf.precedence_over_environment_variables = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
	ret = starpu_init(&conf);
	if (ret == -ENODEV) goto enodev;

	/* Initialize path and name */
	/* register swap disk */
	int new_dd = starpu_disk_register(ops, (void *) base, STARPU_DISK_SIZE_MIN);
	/* can't write on /tmp/ */
	if (new_dd == -ENOENT) goto enoent;

	unsigned int i, j;

	/* Initialize twice as much data as available memory */
	for (i = 0; i < NDATA; i++)
	{
		vector_data_register(&handles[i], -1, 0, (MEMSIZE*1024*1024*2) / NDATA, sizeof(char));
		starpu_task_insert(&zero_cl, STARPU_W, handles[i], 0);
	}
	memset(values, 0, sizeof(values));

	for (i = 0; i < NITER; i++)
	{
		j = rand()%NDATA;
		starpu_task_insert(&inc_cl, STARPU_RW, handles[j], STARPU_VALUE, &j, sizeof(j), 0);
	}

	/* Check and free data */
	for (i = 0; i < NDATA; i++)
	{
		starpu_task_insert(&check_cl, STARPU_R, handles[i], STARPU_VALUE, &i, sizeof(i), 0);
		starpu_data_unregister(handles[i]);
	}

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	return EXIT_SUCCESS;

enoent:
	FPRINTF(stderr, "Couldn't write data: ENOENT\n");
	starpu_shutdown();
enodev:
	return STARPU_TEST_SKIPPED;
}

static int merge_result(int old, int new)
{
	if (new == EXIT_FAILURE)
		return EXIT_FAILURE;
	if (old == 0)
		return 0;
	return new;
}

int main(void)
{
	int ret = 0;
	int ret2;
	char s[128];
	char *ptr;

#ifdef STARPU_HAVE_SETENV
	setenv("STARPU_CALIBRATE_MINIMUM", "1", 1);
#endif

	snprintf(s, sizeof(s), "/tmp/%s-disk-XXXXXX", getenv("USER"));
	ptr = _starpu_mkdtemp(s);
	if (!ptr)
	{
		FPRINTF(stderr, "Cannot make directory '%s'\n", s);
		return STARPU_TEST_SKIPPED;
	}

	setenv("STARPU_LIMIT_CPU_MEM", MEMSIZE_STR, 1);

	/* Build an vector-like interface which doesn't have the any_to_any helper, to force making use of pack/unpack */
	any_to_any = starpu_interface_vector_ops.copy_methods->any_to_any;
	memcpy(&starpu_interface_my_vector_ops, &starpu_interface_vector_ops, sizeof(starpu_interface_my_vector_ops));
	starpu_interface_my_vector_ops.copy_methods = &my_vector_copy_data_methods_s;

	ret = merge_result(ret, dotest(&starpu_disk_stdio_ops, s, starpu_vector_data_register, "Stdio with read/write vector ops"));
	ret = merge_result(ret, dotest(&starpu_disk_stdio_ops, s, starpu_my_vector_data_register, "Stdio with pack/unpack vector ops"));
	ret = merge_result(ret, dotest(&starpu_disk_unistd_ops, s, starpu_vector_data_register, "unistd with read/write vector ops"));
	ret = merge_result(ret, dotest(&starpu_disk_unistd_ops, s, starpu_my_vector_data_register, "unistd with pack/unpack vector ops"));
#ifdef STARPU_LINUX_SYS
	ret = merge_result(ret, dotest(&starpu_disk_unistd_o_direct_ops, s, starpu_vector_data_register, "unistd_direct with read/write vector ops"));
	ret = merge_result(ret, dotest(&starpu_disk_unistd_o_direct_ops, s, starpu_my_vector_data_register, "unistd_direct with pack/unpack vector ops"));
#endif

	ret2 = rmdir(s);
	STARPU_CHECK_RETURN_VALUE(ret2, "rmdir '%s'\n", s);

	return ret;
}
#endif
