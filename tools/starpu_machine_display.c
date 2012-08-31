/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2012  Université de Bordeaux 1
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
#include <config.h>
#include <stdio.h>
#include <starpu.h>

#define PROGNAME "starpu_machine_display"

static void display_worker_names(enum starpu_archtype type)
{
	unsigned nworkers = starpu_worker_get_count_by_type(type);

	int ids[nworkers];
	starpu_worker_get_ids_by_type(type, ids, nworkers);

	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		char name[256];
		starpu_worker_get_name(ids[i], name, 256);
		fprintf(stdout, "\t\t%s\n", name);
	}
}

static void display_combined_worker(unsigned workerid)
{
	int worker_size;
	int *combined_workerid;
	starpu_combined_worker_get_description(workerid, &worker_size, &combined_workerid);

	fprintf(stdout, "\t\t");

	int i;
	for (i = 0; i < worker_size; i++)
	{
		char name[256];

		starpu_worker_get_name(combined_workerid[i], name, 256);

		fprintf(stdout, "%s\t", name);
	}

	fprintf(stdout, "\n");
}

static void display_all_combined_workers(void)
{
	unsigned ncombined_workers = starpu_combined_worker_get_count();

	if (ncombined_workers == 0)
		return;

	unsigned nworkers = starpu_worker_get_count();

	fprintf(stdout, "\t%d Combined workers\n", ncombined_workers);

	unsigned i;
	for (i = 0; i < ncombined_workers; i++)
		display_combined_worker(nworkers + i);
}

static void parse_args(int argc, char **argv, int *force)
{
	int i;

	if (argc == 1)
		return;

	for (i = 1; i < argc; i++)
	{
		if (strncmp(argv[i], "--force", 7) == 0 || strncmp(argv[i], "-f", 2) == 0)
		{
			*force = 1;
		}
		else if (strncmp(argv[i], "--help", 6) == 0 || strncmp(argv[i], "-h", 2) == 0)
		{
			(void) fprintf(stderr, "\
Show the processing units that StarPU can use, and the	      \n	\
bandwitdh and affinity measured between the memory nodes.     \n	\
                                                              \n	\
Usage: %s [OPTION]                                            \n	\
                                                              \n	\
Options:                                                      \n	\
	-h, --help       display this help and exit           \n	\
	-v, --version    output version information and exit  \n	\
	-f, --force      force bus sampling and show measures \n	\
                                                              \n	\
Report bugs to <" PACKAGE_BUGREPORT ">.\n",
PROGNAME);
			exit(EXIT_FAILURE);
		}
		else if (strncmp(argv[i], "--version", 9) == 0 || strncmp(argv[i], "-v", 2) == 0)
		{
			(void) fprintf(stderr, "%s %d.%d\n",
				       PROGNAME, STARPU_MAJOR_VERSION, STARPU_MINOR_VERSION);
			exit(EXIT_FAILURE);
		}
		else
		{
			fprintf(stderr, "Unknown arg %s\n", argv[1]);
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char **argv)
{
	int force = 0;
	struct starpu_conf conf;

	parse_args(argc, argv, &force);

	starpu_conf_init(&conf);
	if (force)
		conf.bus_calibrate = 1;

	/* Even if starpu_init returns -ENODEV, we should go on : we will just
	 * print that we found no device. */
	(void) starpu_init(&conf);

	unsigned ncpu = starpu_cpu_worker_get_count();
	unsigned ncuda = starpu_cuda_worker_get_count();
	unsigned nopencl = starpu_opencl_worker_get_count();

	fprintf(stdout, "StarPU has found :\n");

	fprintf(stdout, "\t%d CPU cores\n", ncpu);
	display_worker_names(STARPU_CPU_WORKER);

	fprintf(stdout, "\t%d CUDA devices\n", ncuda);
	display_worker_names(STARPU_CUDA_WORKER);

	fprintf(stdout, "\t%d OpenCL devices\n", nopencl);
	display_worker_names(STARPU_OPENCL_WORKER);

	display_all_combined_workers();

	fprintf(stdout, "\nbandwidth ...\n");
	starpu_bus_print_bandwidth(stdout);

	starpu_shutdown();

	return 0;
}
