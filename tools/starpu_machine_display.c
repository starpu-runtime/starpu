/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/utils.h>

#define PROGNAME "starpu_machine_display"

static void usage()
{
	fprintf(stderr, "Show the processing units that StarPU can use,\n");
	fprintf(stderr, "and the bandwitdh and affinity measured between the memory nodes.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Usage: %s [OPTION]\n", PROGNAME);
	fprintf(stderr, "\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-h, --help          display this help and exit\n");
	fprintf(stderr, "\t-v, --version       output version information and exit\n");
	fprintf(stderr, "\t-i, --info          display the name of the files containing the information\n");
	fprintf(stderr, "\t-f, --force         force bus sampling and show measures \n");
	fprintf(stderr, "\t-w, --worker <type> only show workers of the given type\n");
	fprintf(stderr, "\t-c, --count         only display the number of workers\n");
	fprintf(stderr, "\t-n, --notopology    do not display the bandwitdh and affinity\n");
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

static void parse_args(int argc, char **argv, int *force, int *info, int *count, int *topology, char **worker_type)
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
		else if (strncmp(argv[i], "--count", 7) == 0 || strncmp(argv[i], "-c", 2) == 0)
		{
			*count = 1;
		}
		else if (strncmp(argv[i], "--worker", 8) == 0 || strncmp(argv[i], "-w", 2) == 0)
		{
			*worker_type = strdup(argv[++i]);
		}
		else if (strncmp(argv[i], "--notopology", 12) == 0 || strncmp(argv[i], "-n", 2) == 0)
		{
			*topology = 0;
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
	int count = 0;
	int topology = 1;
	char *worker_type = NULL;
	struct starpu_conf conf;

	parse_args(argc, argv, &force, &info, &count, &topology, &worker_type);

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
	starpu_worker_wait_for_initialisation();

	if (info)
	{
		starpu_bus_print_filenames(stdout);
		starpu_shutdown();
		return 0;
	}

	char real_hostname[128];
	char starpu_hostname[128];
	gethostname(real_hostname, sizeof(real_hostname));
	_starpu_gethostname(starpu_hostname, sizeof(starpu_hostname));
	fprintf(stdout, "Real hostname: %s (StarPU hostname: %s)\n", real_hostname, starpu_hostname);

	const char *env[] =
	{
		"STARPU_NCPU",
		"STARPU_NCPUS",
		"STARPU_NCUDA",
		"STARPU_NHIP",
		"STARPU_NOPENCL",
		"STARPU_NMAX_FPGA",
		"STARPU_NMPI_MS",
		"STARPU_NTCPIP_MS",

		"STARPU_WORKERS_CPUID",
		"STARPU_WORKERS_COREID",
		"STARPU_NTHREADS_PER_CORE",
		"STARPU_RESERVE_NCPU",
		"STARPU_MAIN_THREAD_BIND",
		"STARPU_MAIN_THREAD_CPUID",
		"STARPU_MAIN_THREAD_COREID",

		"STARPU_WORKERS_CUDAID",
		"STARPU_CUDA_THREAD_PER_WORKER",
		"STARPU_CUDA_THREAD_PER_DEV",

		"STARPU_WORKERS_OPENCLID",
		"STARPU_WORKERS_MAX_FPGAID",

		"STARPU_MPI_MS_MULTIPLE_THREAD",
		"STARPU_NMPIMSTHREADS",
		"STARPU_TCPIP_MS_MULTIPLE_THREAD",
		"STARPU_NTCPIPMSTHREADS",

		"STARPU_MPI_HOSTNAMES",
		"STARPU_HOSTNAME",
		NULL
	};

	int i;
	static int message=0;
	for (i = 0; env[i]; i++)
	{
		const char *e = getenv(env[i]);
		if (e)
		{
			if (!message)
			{
				fprintf(stdout, "Environment variables\n");
				message=1;
			}
			fprintf(stdout, "\t%s=%s\n", env[i], e);
		}
	}
	if (message)
		fprintf(stdout,"\n");

	void (*func)(FILE *output, enum starpu_worker_archtype type) = &starpu_worker_display_names;
	if (count == 1)
		func = &starpu_worker_display_count;

	if (worker_type)
	{
		if (strcmp(worker_type, "CPU") == 0)
			func(stdout, STARPU_CPU_WORKER);
		else if (strcmp(worker_type, "CUDA") == 0)
			func(stdout, STARPU_CUDA_WORKER);
		else if (strcmp(worker_type, "OpenCL") == 0)
			func(stdout, STARPU_OPENCL_WORKER);
		else if (strcmp(worker_type, "HIP") == 0)
			func(stdout, STARPU_HIP_WORKER);
#ifdef STARPU_USE_MPI_MASTER_SLAVE
		else if (strcmp(worker_type, "MPI_MS") == 0)
			func(stdout, STARPU_MPI_MS_WORKER);
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
		else if (strcmp(worker_type, "TCPIP_MS") == 0)
			func(stdout, STARPU_TCPIP_MS_WORKER);
#endif
		else
			fprintf(stderr, "Unknown worker type '%s'\n", worker_type);
	}
	else
	{
		fprintf(stdout, "StarPU has found :\n");

		func(stdout, STARPU_CPU_WORKER);
		func(stdout, STARPU_CUDA_WORKER);
		func(stdout, STARPU_OPENCL_WORKER);
		func(stdout, STARPU_HIP_WORKER);
#ifdef STARPU_USE_MPI_MASTER_SLAVE
		func(stdout, STARPU_MPI_MS_WORKER);
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
		func(stdout, STARPU_TCPIP_MS_WORKER);
#endif

		display_all_combined_workers();
	}

	if (ret != -ENODEV)
	{
		if (topology == 1)
		{
			fprintf(stdout, "\ntopology ... (hwloc logical indexes)\n");
			starpu_topology_print(stdout);

			fprintf(stdout, "\nbandwidth (MB/s) and latency (us)...\n");
			starpu_bus_print_bandwidth(stdout);
		}
		starpu_shutdown();
	}

	return 0;
}
