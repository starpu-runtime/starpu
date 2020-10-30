/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * Try various values for STARPU_WORKERS_CPUID, checking that the
 * expected binding does happen
 */

#if !defined(STARPU_USE_CPU) || !defined(STARPU_HAVE_HWLOC) || !defined(STARPU_HAVE_SETENV)
#warning no cpu are available. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

#include <hwloc.h>

#ifdef STARPU_QUICK_CHECK
#define CPUSTEP 8
#define NB_TESTS 1
#else
#define CPUSTEP 1
#define NB_TESTS 5
#endif

int nhwpus;
long workers_cpuid[STARPU_NMAXWORKERS];
int workers_id[STARPU_NMAXWORKERS];

static int check_workers_mapping(long *cpuid, int *workerids, int nb_workers)
{
	int i;
	for (i=0; i<nb_workers; i++)
	{
		int workerid = workerids[i];
		long bindid = starpu_worker_get_bindid(workerid);
		if ( bindid != cpuid[i])
		{
			fprintf(stderr, "Worker %d (%s) is on cpu %ld rather than on %ld\n",
			       workerid, starpu_worker_get_type_as_string(starpu_worker_get_type(workerid)),
			       bindid, cpuid[i]);
			return 0;
		}
	}
	return 1;
}

static void copy_cpuid_array(long *dst, long *src, unsigned n)
{
	int i;

	memcpy(dst, src, n * sizeof(long));
	for (i=n; i<STARPU_NMAXWORKERS; i++)
		dst[i] = src[i%n];
}

static char *array_to_str(long *array, int n)
{
	int i;
	int len = n * 3 * sizeof(long);
	char *str = malloc(len);
	char *ptr = str;

	for (i=0; i<n; i++)
	{
		int nchar;
		nchar = snprintf(ptr, len - (ptr-str), "%ld ", array[i]);
		ptr += nchar;
	}

	return str;
}

static int test_combination(long *combination, unsigned n)
{
	int ret, device_workers;
	char *str;

	copy_cpuid_array(workers_cpuid, combination, n);

	str = array_to_str(workers_cpuid, n);
	setenv("STARPU_WORKERS_CPUID", str, 1);
	free(str);

	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.nmic = 0;
	conf.nmpi_ms = 0;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	device_workers = 0;

	/* Check for all cpus */
	{
		int nb_workers;
		nb_workers = starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, workers_id, starpu_worker_get_count());
		if (!check_workers_mapping(workers_cpuid + device_workers, workers_id, nb_workers))
			return -1;
	}

	starpu_shutdown();
	return 1;
}

static long * generate_arrangement(int arr_size, long *set, int set_size)
{
	int i;

	STARPU_ASSERT(arr_size <= set_size);

	for (i=0; i<arr_size; i++)
	{
		/* Pick a random value in the set */
		int j = starpu_lrand48() % (set_size - i);

		/* Switch the value picked up with the beginning value of set */
		long tmp = set[i+j];
		set[i] = set[i+j];
		set[i+j] = tmp;
	}

	return set;
}

static void init_array(long *a, int n)
{
	int i;

	for(i=0; i<n; i++)
		a[i] = i;
}

int main(void)
{
	int i;
	long *cpuids;
	hwloc_topology_t topology;

	hwloc_topology_init(&topology);
	hwloc_topology_load(topology);
	nhwpus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
	if (nhwpus > STARPU_NMAXWORKERS)
		nhwpus = STARPU_NMAXWORKERS;

	for (i=0; i<STARPU_NMAXWORKERS; i++)
		workers_id[i] = -1;

	cpuids = malloc(nhwpus * sizeof(long));

	/* Evaluate several random values of STARPU_WORKERS_CPUID
	 * and check mapping for each one
	 */
	for (i=1; i<=nhwpus; i += CPUSTEP)
	{
		int j;
		for (j=0; j<NB_TESTS; j++)
		{
			int ret;

			init_array(cpuids, nhwpus);
			generate_arrangement(i, cpuids, nhwpus);
			ret = test_combination(cpuids, i);
			if (ret == STARPU_TEST_SKIPPED) return STARPU_TEST_SKIPPED;
			if (ret != 1)
			{
				free(cpuids);
				hwloc_topology_destroy(topology);
				return EXIT_FAILURE;
			}
		}
	}

	free(cpuids);

	hwloc_topology_destroy(topology);

	return EXIT_SUCCESS;
}
#endif
