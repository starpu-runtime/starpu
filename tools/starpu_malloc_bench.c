/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2026-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>
#include <sys/mman.h>
#include <common/config.h>
#include <common/utils.h>
#include <common/timing.h>

#define PROGNAME "starpu_malloc_bench"

#ifndef MAP_HUGE_2MB
/* As suggested in mmap(2) */
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

struct params
{
	size_t size_min;
	size_t size_inc;
	size_t size_max;
	int niter;
	int pin_only;
	int alloc_only;
	int huge_pages;
	int manual_prefault;
	int mmap_flags;
	int starpu_malloc;
};

static struct params params;

static void usage(void)
{
	printf("Usage: starpu_malloc_bench [OPTION...]\n");
	printf("Measure the cost of memory allocation and/or pinning according to buffer size.\n");
	printf("Options:\n");
	printf("      --alloc-only         measure the cost of the memory allocation alone,\n");
	printf("                           without the memory pinning\n");
	printf("  -h, --help               print this message\n");
	printf("      --huge-pages         use 2MB pages (aka huge pages) rather than\n");
	printf("                           default 4KB pages\n");
	printf("      --manual-prefault    manually prefault newly allocated memory pages, by default\n");
	printf("                           prefault is done by providing the MAP_POPULATE flag to mmap(2)\n");
	printf("      --niter       NITER  set number of iteration per size value to NITER (default %d)\n", params.niter);
	printf("      --pin-only           measure the cost of the memory pinning alone,\n");
	printf("                           without memory allocation\n");
	printf("      --size-min    SIZE   set minimum size to SIZE (default %zu)\n", params.size_min);
	printf("      --size-inc    SIZE   set size increment to SIZE (default %zu)\n", params.size_inc);
	printf("      --size-max    SIZE   set maximum size to SIZE (default %zu)\n", params.size_max);
	printf("      --starpu_malloc      use the `starpu_malloc' function for allocating (and pinning\n");
	printf("                           if any GPU worker is available) memory, incompatible with\n");
	printf("                           --alloc-only, --huge-pages, and --pin-only\n");
	printf("      --version            print version\n");
	printf("Notes:\n");
	printf("- Multiple GPUs mess up the benchmark. If not set externally, this\n");
	printf("  program sets STARPU_NCPU and STARPU_NCUDA to 1.\n");
}

static void parse_args(int argc, char **argv)
{
	int i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i], "--alloc-only") == 0)
		{
			params.alloc_only = 1;
			i += 1;
		}
		else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0))
		{
			usage();
			exit(EXIT_SUCCESS);
		}
		else if (strcmp(argv[i], "--huge-pages") == 0)
		{
			params.huge_pages = 1;
			i += 1;
		}
		else if (strcmp(argv[i], "--manual-prefault") == 0)
		{
			params.manual_prefault = 1;
			i += 1;
		}
		else if (strcmp(argv[i], "--niter") == 0)
		{
			STARPU_ASSERT_MSG(i + 1 < argc, "missing option argument");
			params.niter = atoi(argv[i+1]);
			i += 2;
		}
		else if (strcmp(argv[i], "--pin-only") == 0)
		{
			params.pin_only = 1;
			i += 1;
		}
		else if (strcmp(argv[i], "--size-min") == 0)
		{
			STARPU_ASSERT_MSG(i + 1 < argc, "missing option argument");
			params.size_min = atol(argv[i+1]);
			i += 2;
		}
		else if (strcmp(argv[i], "--size-inc") == 0)
		{
			STARPU_ASSERT_MSG(i + 1 < argc, "missing option argument");
			params.size_inc = atol(argv[i+1]);
			i += 2;
		}
		else if (strcmp(argv[i], "--size-max") == 0)
		{
			STARPU_ASSERT_MSG(i + 1 < argc, "missing option argument");
			params.size_max = atol(argv[i+1]);
			i += 2;
		}
		else if (strcmp(argv[i], "--starpu-malloc") == 0)
		{
			params.starpu_malloc = 1;
			i += 1;
		}
		else if (strcmp(argv[i], "--version") == 0)
		{
			printf("%s (%s) %s\n", PROGNAME, PACKAGE_NAME, PACKAGE_VERSION);
			exit(EXIT_SUCCESS);
		}
		else
		{
			fprintf(stderr, "error: unknown argument %s\n", argv[i]);
			exit(EXIT_FAILURE);
		}
	}
}

/* Prefault the memory pages between _PTR and (_PTR + SIZE). Doing it manually
   avoids caveats of MAP_POPULATE mentioned in mmap(2). */
static void prefault(void *_ptr, size_t size)
{
	char *ptr = _ptr;
	size_t page_size = getpagesize();
	if (params.huge_pages)
		page_size = 2 * 1024 * 1024;
	for (char *i = ptr; i < ptr + size; i += page_size)
		*i = '\0';
}

/* Implementation of bench_size_index using mmap(2) and starpu_memory pin. */
static void bench_size_index_starpu_memory_pin(size_t size, int index)
{
	struct timespec t0;
	struct timespec t1;
	struct timespec t2;
	int ret;
	double dt;
	void *ptr;

	_starpu_clock_gettime(&t0);

	ptr = mmap(NULL,
		   size,
		   PROT_READ | PROT_WRITE,
		   params.mmap_flags,
		   -1,
		   0);
	STARPU_ASSERT(ptr != MAP_FAILED);
	if (params.manual_prefault)
		prefault(ptr, size);

	_starpu_clock_gettime(&t1);

	ret = starpu_memory_pin(ptr, size);

	_starpu_clock_gettime(&t2);

	STARPU_ASSERT(ret == 0);
	if (params.alloc_only)
		dt = starpu_timing_timespec_delay_us(&t0, &t1);
	else if (params.pin_only)
		dt = starpu_timing_timespec_delay_us(&t1, &t2);
	else
		dt = starpu_timing_timespec_delay_us(&t0, &t2);

	printf("%d\t%zu\t%.2e\n", index, size, dt);

	ret = starpu_memory_unpin(ptr, size);
	STARPU_ASSERT(ret == 0);
	ret = munmap(ptr, size);
	STARPU_ASSERT(ret == 0);
}

/* Implementation of bench_size_index using starpu_malloc. */
static void bench_size_index_starpu_malloc(size_t size, int index)
{
	struct timespec t0;
	struct timespec t1;
	int ret;
	double dt;
	void *ptr;

	_starpu_clock_gettime(&t0);

	ret = starpu_malloc(&ptr, size);
	STARPU_ASSERT(ret == 0);

	if (params.manual_prefault)
		prefault(ptr, size);

	_starpu_clock_gettime(&t1);

	dt = starpu_timing_timespec_delay_us(&t0, &t1);

	printf("%d\t%zu\t%.2e\n", index, size, dt);

	ret = starpu_free_noflag(ptr, size);
	STARPU_ASSERT(ret == 0);
}

/* Perform one iteration of the benchmark. */
static void bench_size_index(size_t size, int index)
{
	if (params.starpu_malloc)
		bench_size_index_starpu_malloc(size, index);
	else
		bench_size_index_starpu_memory_pin(size, index);
}

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	int ret;

	memset(&params, 0, sizeof(params));
	params.size_min   = 131072;
	params.size_inc   = params.size_min;
	params.size_max   = 4096 * 1024;
	params.niter      = 16;
	params.mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;

	parse_args(argc, argv);

	starpu_conf_init(&conf);
	conf.ncpus = 1;
	conf.ncuda = 1;
	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_ASSERT(ret == 0);

	if (params.alloc_only && params.pin_only)
	{
		fprintf(stderr,
			"%s: error: options --alloc-only and --pin-only both given\n",
			argv[0]);
		exit(EXIT_FAILURE);
	}

	if (params.starpu_malloc)
	{
		if (params.alloc_only || params.huge_pages || params.pin_only)
		{
			fprintf(stderr,
				"%s: error: option incompatible with --starpu-malloc was given\n",
				argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	if (!params.manual_prefault)
		params.mmap_flags |= MAP_POPULATE;
	if (params.huge_pages)
		params.mmap_flags |= MAP_HUGETLB | MAP_HUGE_2MB;

	printf("Index\tSize\tTime (us)\n");

	int c = 0; /* index for size loop */
	for (size_t size = params.size_min; size <= params.size_max; size += params.size_inc)
	{
		for (int i = 0; i < params.niter; i++)
		{
			int index = c*params.niter + i;
			bench_size_index(size, index);
		}
		c++;
	}

	starpu_shutdown();
	return EXIT_SUCCESS;
}
