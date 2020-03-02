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

#include <stdio.h>
#include <starpu.h>
#include <starpu_scheduler.h>
#include <common/config.h>

#define PROGNAME "starpu_machine_display"

static void usage()
{
	fprintf(stderr, "Show the processing units that StarPU can use,\n");
	fprintf(stderr, "and the bandwitdh and affinity measured between the memory nodes.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Usage: %s [OPTION]\n", PROGNAME);
	fprintf(stderr, "\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-h, --help       display this help and exit\n");
	fprintf(stderr, "\t-v, --version    output version information and exit\n");
	fprintf(stderr, "\t-i, --info       display the name of the files containing the information\n");
	fprintf(stderr, "\t-f, --force      force bus sampling and show measures \n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Report bugs to <%s>.\n", PACKAGE_BUGREPORT);
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

	fprintf(stdout, "\t%u Combined workers\n", ncombined_workers);

	unsigned i;
	for (i = 0; i < ncombined_workers; i++)
		display_combined_worker(nworkers + i);
}

static void parse_args(int argc, char **argv, int *force, int *info)
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
		else if (strncmp(argv[i], "--info", 6) == 0 || strncmp(argv[i], "-i", 2) == 0)
		{
			*info = 1;
		}
		else if (strncmp(argv[i], "--help", 6) == 0 || strncmp(argv[i], "-h", 2) == 0)
		{
			usage();
			exit(EXIT_FAILURE);
		}
		else if (strncmp(argv[i], "--version", 9) == 0 || strncmp(argv[i], "-v", 2) == 0)
		{
			fputs(PROGNAME " (" PACKAGE_NAME ") " PACKAGE_VERSION "\n", stderr);
			exit(EXIT_FAILURE);
		}
		else
		{
			fprintf(stderr, "Unknown arg %s\n", argv[1]);
			usage();
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char **argv)
{
	int ret;
	int force = 0;
	int info = 0;
	struct starpu_conf conf;

	parse_args(argc, argv, &force, &info);

	starpu_conf_init(&conf);
	if (force)
		conf.bus_calibrate = 1;

	/* Even if starpu_init returns -ENODEV, we should go on : we will just
	 * print that we found no device. */
	ret = starpu_init(&conf);
	if (ret != 0 && ret != -ENODEV)
	{
		return ret;
	}

	if (info)
	{
		starpu_bus_print_filenames(stdout);
		starpu_shutdown();
		return 0;
	}

	fprintf(stdout, "StarPU has found :\n");

	starpu_worker_display_names(stdout, STARPU_CPU_WORKER);
	starpu_worker_display_names(stdout, STARPU_CUDA_WORKER);
	starpu_worker_display_names(stdout, STARPU_OPENCL_WORKER);
#ifdef STARPU_USE_MIC
	starpu_worker_display_names(stdout, STARPU_MIC_WORKER);
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	starpu_worker_display_names(stdout, STARPU_MPI_MS_WORKER);
#endif

	display_all_combined_workers();

	if (ret != -ENODEV)
	{
		fprintf(stdout, "\ntopology ... (hwloc logical indexes)\n");
		starpu_topology_print(stdout);

		fprintf(stdout, "\nbandwidth (MB/s) and latency (us)...\n");
		starpu_bus_print_bandwidth(stdout);

		starpu_shutdown();
	}

	return 0;
}
