/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include "../helper.h"

/*
 * Try to write into disk memory
 * Use mechanism to push datas from main ram to disk ram
 * Here we make copies between buffers, that StarPU has to evict while
 * progressing because there is not enough room for all of them.
 */

/* RAM is not enough to hold 6 times NX
 * DISK is just enough to hold 6 times NX */

/* size of one vector */
#ifdef STARPU_QUICK_CHECK
#  define	RAM	"1"
#  define	DISK	64
#  define	NX	(256*1024/sizeof(double))
#else
#  define	NX	(32*1048576/sizeof(double))
#  define	RAM	"160"
#  define	DISK	200
#endif

#if !defined(STARPU_HAVE_SETENV)
#warning setenv is not defined. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

int dotest(struct starpu_disk_ops *ops, void *param)
{
	double *A,*B,*C,*D,*E,*F;
	int ret;

	/* limit main ram to force to push in disk */
	setenv("STARPU_LIMIT_CPU_NUMA_MEM", RAM, 1);

	/* Initialize StarPU without GPU devices to make sure the memory of the GPU devices will not be used */
	// Ignore environment variables as we want to force the exact number of workers
	struct starpu_conf conf;
	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return EXIT_FAILURE;
	conf.precedence_over_environment_variables = 1;
	conf.ncpus = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
	ret = starpu_init(&conf);
	if (ret == -ENODEV) goto enodev;

	/* register a disk */
	int new_dd = starpu_disk_register(ops, param, 1024*1024*DISK);
	/* can't write on /tmp/ */
	if (new_dd == -ENOENT) goto enoent;

	/* allocate two memory spaces */
	starpu_malloc_flags((void **)&A, NX*sizeof(double), STARPU_MALLOC_COUNT);
	starpu_malloc_flags((void **)&F, NX*sizeof(double), STARPU_MALLOC_COUNT);

	FPRINTF(stderr, "TEST DISK MEMORY \n");

	unsigned int j;
	/* initialization with bad values */
	for(j = 0; j < NX; ++j)
	{
		A[j] = j;
		F[j] = -j;
	}

	starpu_data_handle_t vector_handleA, vector_handleB, vector_handleC, vector_handleD, vector_handleE, vector_handleF;

	/* register vector in starpu */
	starpu_vector_data_register(&vector_handleA, STARPU_MAIN_RAM, (uintptr_t)A, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleB, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleC, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleD, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleE, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleF, STARPU_MAIN_RAM, (uintptr_t)F, NX, sizeof(double));

	/* copy vector A->B, B->C... */
	starpu_data_cpy(vector_handleB, vector_handleA, 0, NULL, NULL);
	starpu_data_cpy(vector_handleC, vector_handleB, 0, NULL, NULL);
	starpu_data_cpy(vector_handleD, vector_handleC, 0, NULL, NULL);
	starpu_data_cpy(vector_handleE, vector_handleD, 0, NULL, NULL);
	starpu_data_cpy(vector_handleF, vector_handleE, 0, NULL, NULL);

	/* StarPU does not need to manipulate the array anymore so we can stop
 	 * monitoring it */

	/* free them */
	starpu_data_unregister(vector_handleA);
	starpu_data_unregister(vector_handleB);
	starpu_data_unregister(vector_handleC);
	starpu_data_unregister(vector_handleD);
	starpu_data_unregister(vector_handleE);
	starpu_data_unregister(vector_handleF);

	/* check if computation is correct */
	int try = 1;
	for (j = 0; j < NX; ++j)
		if (A[j] != F[j])
		{
			FPRINTF(stderr, "Fail A %f != F %f \n", A[j], F[j]);
			try = 0;
		}

	starpu_free_flags(A, NX*sizeof(double), STARPU_MALLOC_COUNT);
	starpu_free_flags(F, NX*sizeof(double), STARPU_MALLOC_COUNT);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	if(try)
		FPRINTF(stderr, "TEST SUCCESS\n");
	else
		FPRINTF(stderr, "TEST FAIL\n");
	return try ? EXIT_SUCCESS : EXIT_FAILURE;

enodev:
	return STARPU_TEST_SKIPPED;
enoent:
	FPRINTF(stderr, "Couldn't write data: ENOENT\n");
	starpu_shutdown();
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
		FPRINTF(stderr, "Cannot make directory <%s>\n", s);
		return STARPU_TEST_SKIPPED;
	}

	ret = merge_result(ret, dotest(&starpu_disk_stdio_ops, s));
	ret = merge_result(ret, dotest(&starpu_disk_unistd_ops, s));
#ifdef STARPU_LINUX_SYS
	ret = merge_result(ret, dotest(&starpu_disk_unistd_o_direct_ops, s));
#endif
#ifdef STARPU_HAVE_HDF5
	ret = merge_result(ret, dotest(&starpu_disk_hdf5_ops, s));
#endif

	ret2 = rmdir(s);
	if (ret2 < 0)
		STARPU_CHECK_RETURN_VALUE(-errno, "rmdir '%s'\n", s);
	return ret;
}
#endif
