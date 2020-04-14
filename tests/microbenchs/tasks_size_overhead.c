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

#include <stdio.h>
#include <unistd.h>

#include <starpu.h>
#include "../helper.h"

/*
 * This benchmark creates a thousand tasks of the same (small) duration, with
 * various number of cpus and various durations.
 *
 * Use ./tasks_size_overhead.sh to generate a plot of the result.
 *
 * Thanks Martin Tillenius for the idea.
 */

#define START 4
#define STOP 4096
#ifdef STARPU_QUICK_CHECK
#define FACTOR 64
#else
#define FACTOR 2
#endif

#ifdef STARPU_QUICK_CHECK
#define CPUSTEP 8
#else
#define CPUSTEP 1
#endif

#ifdef STARPU_QUICK_CHECK
static unsigned ntasks = 1;
#elif !defined(STARPU_LONG_CHECK)
static unsigned ntasks = 64;
#else
static unsigned ntasks = 256;
#endif

static unsigned nbuffers = 0;
static unsigned total_nbuffers = 0;

static unsigned mincpus = 1, maxcpus, cpustep = CPUSTEP;
static unsigned mintime = START, maxtime = STOP, factortime = FACTOR;

struct starpu_task *tasks;

void func(void *descr[], void *arg)
{
	(void)descr;
	unsigned n = (uintptr_t)arg;
	long usec = 0;
	double tv1 = starpu_timing_now();
	do
	{
		double tv2 = starpu_timing_now();
		usec = tv2 - tv1;
	}
	while (usec < n);
}

double cost_function(struct starpu_task *t, struct starpu_perfmodel_arch *a, unsigned i)
{
	(void) t; (void) i; (void) a;
	unsigned n = (uintptr_t) t->cl_arg;
	return n;
}

static struct starpu_perfmodel perf_model =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = cost_function,
};

static struct starpu_codelet codelet =
{
	.cpu_funcs = {func},
	.nbuffers = 0,
	.modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R},
	.model = &perf_model,
};

static void parse_args(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "i:b:B:c:C:s:t:T:f:h")) != -1)
	switch(c)
	{
		case 'i':
			ntasks = atoi(optarg);
			break;
		case 'b':
			nbuffers = atoi(optarg);
			codelet.nbuffers = nbuffers;
			break;
		case 'B':
			total_nbuffers = atoi(optarg);
			break;
		case 'c':
			mincpus = atoi(optarg);
			break;
		case 'C':
			maxcpus = atoi(optarg);
			break;
		case 's':
			cpustep = atoi(optarg);
			break;
		case 't':
			mintime = atoi(optarg);
			break;
		case 'T':
			maxtime = atoi(optarg);
			break;
		case 'f':
			factortime = atoi(optarg);
			break;
		case 'h':
			fprintf(stderr, "\
Usage: %s [-h]\n\
          [-i ntasks] [-b nbuffers] [-B total_nbuffers]\n\
          [-c mincpus] [ -C maxcpus] [-s cpustep]\n\
	  [-t mintime] [-T maxtime] [-f factortime]\n\n", argv[0]);
			fprintf(stderr,"\
runs 'ntasks' tasks\n\
- using 'nbuffers' data each, randomly among 'total_nbuffers' choices,\n\
- with varying task durations, from 'mintime' to 'maxtime' (using 'factortime')\n\
- on varying numbers of cpus, from 'mincpus' to 'maxcpus' (using 'cpustep')\n\
\n\
currently selected parameters: %u tasks using %u buffers among %u, from %uus to %uus (factor %u), from %u cpus to %u cpus (step %u)\n\
", ntasks, nbuffers, total_nbuffers, mintime, maxtime, factortime, mincpus, maxcpus, cpustep);
			exit(EXIT_SUCCESS);
			break;
	}
}

int main(int argc, char **argv)
{
	int ret;
	unsigned i;
	unsigned size;
	unsigned ncpus;

	double timing;
	double start;
	double end;

	struct starpu_conf conf;

	unsigned buffer;
	char *starpu_sched = getenv("STARPU_SCHED");

	/* Get number of CPUs */
	starpu_conf_init(&conf);
	conf.ncuda = 0;
	conf.nopencl = 0;
#ifdef STARPU_SIMGRID
	/* This will get serialized, avoid spending too much time on it. */
	maxcpus = 2;
#else
	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
	maxcpus = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
	starpu_shutdown();
#endif

#ifdef STARPU_HAVE_UNSETENV
	/* That was useful to force the max number of cpus to use, but now we
	 * want to make it vary */
	unsetenv("STARPU_NCPUS");
	unsetenv("STARPU_NCPU");
#endif

	if (STARPU_RUNNING_ON_VALGRIND)
	{
		factortime *= 4;
		cpustep *= 4;
	}

	parse_args(argc, argv);

	float *buffers[total_nbuffers?total_nbuffers:1];

	/* Allocate data */
	for (buffer = 0; buffer < total_nbuffers; buffer++)
		buffers[buffer] = (float *) calloc(16, sizeof(float));

	tasks = (struct starpu_task *) calloc(1, ntasks*maxcpus*sizeof(struct starpu_task));

	/* Emit headers and compute raw tasks speed */
	FPRINTF(stdout, "# tasks : %u buffers : %u total_nbuffers : %u\n", ntasks, nbuffers, total_nbuffers);
	FPRINTF(stdout, "# ncpus\t");
	for (size = mintime; size <= maxtime; size *= factortime)
		FPRINTF(stdout, "%u iters(us)\ttotal(s)\t", size);
	FPRINTF(stdout, "\n");
	FPRINTF(stdout, "\"seq\"\t");
	for (size = mintime; size <= maxtime; size *= factortime)
	{
		double dstart, dend;
		dstart = starpu_timing_now();
		for (i = 0; i < ntasks; i++)
			func(NULL, (void*) (uintptr_t) size);
		dend = starpu_timing_now();
		FPRINTF(stdout, "%.0f       \t%f\t", (dend-dstart)/ntasks, (dend-dstart)/1000000);
	}
	FPRINTF(stdout, "\n");
	fflush(stdout);

	starpu_data_handle_t data_handles[total_nbuffers?total_nbuffers:1];

	if (nbuffers && !total_nbuffers)
	{
		fprintf(stderr,"can not have %u buffers with %u total buffers\n", nbuffers, total_nbuffers);
		goto error;
	}

	/* For each number of cpus, benchmark */
	for (ncpus= mincpus; ncpus <= maxcpus; ncpus += cpustep)
	{
		FPRINTF(stdout, "%u\t", ncpus);
		fflush(stdout);

		conf.ncpus = ncpus;
		ret = starpu_init(&conf);
		if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

		for (buffer = 0; buffer < total_nbuffers; buffer++)
			starpu_vector_data_register(&data_handles[buffer], STARPU_MAIN_RAM, (uintptr_t)buffers[buffer], 16, sizeof(float));

		for (size = mintime; size <= maxtime; size *= factortime)
		{
			/* submit tasks */
			start = starpu_timing_now();
			for (i = 0; i < ntasks * ncpus; i++)
			{
				starpu_data_handle_t *handles;
				starpu_task_init(&tasks[i]);
				tasks[i].callback_func = NULL;
				tasks[i].cl = &codelet;
				tasks[i].cl_arg = (void*) (uintptr_t) size;
				tasks[i].synchronous = 0;

				if (nbuffers > STARPU_NMAXBUFS)
				{
					tasks[i].dyn_handles = malloc(nbuffers * sizeof(*data_handles));
					handles = tasks[i].dyn_handles;
					tasks[i].dyn_modes = malloc(nbuffers * sizeof(tasks[i].dyn_modes[0]));
					for (buffer = 0; buffer < nbuffers; buffer++)
						tasks[i].dyn_modes[buffer] = STARPU_R;
				}
				else
					handles = tasks[i].handles;

				if (nbuffers >= total_nbuffers)
					for (buffer = 0; buffer < nbuffers; buffer++)
						handles[buffer] = data_handles[buffer%total_nbuffers];
				else
					for (buffer = 0; buffer < nbuffers; buffer++)
						handles[buffer] = data_handles[starpu_lrand48()%total_nbuffers];

				ret = starpu_task_submit(&tasks[i]);
				if (ret == -ENODEV) goto enodev;
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task");
			}
			ret = starpu_task_wait_for_all();
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");
			end = starpu_timing_now();

			for (i = 0; i < ntasks * ncpus; i++)
				starpu_task_clean(&tasks[i]);

			timing = end - start;

			FPRINTF(stdout, "%u\t%f\t", size, timing/ncpus/1000000);
			fflush(stdout);

			{
				char *output_dir = getenv("STARPU_BENCH_DIR");
				char *bench_id = getenv("STARPU_BENCH_ID");
				char *sched = getenv("STARPU_SCHED");

				if (output_dir && bench_id)
				{
					char file[1024];
					FILE *f;

					snprintf(file, sizeof(file), "%s/tasks_size_overhead_total%s%s.dat", output_dir, sched?"_":"", sched?sched:"");
					f = fopen(file, "a");
					fprintf(f, "%s\t%u\t%u\t%f\n", bench_id, ncpus, size, timing/1000000 /(ntasks*ncpus) *1000);
					fclose(f);
				}
			}
		}

		for (buffer = 0; buffer < total_nbuffers; buffer++)
		{
			starpu_data_unregister(data_handles[buffer]);
		}

		starpu_shutdown();

		FPRINTF(stdout, "\n");
		fflush(stdout);
	}

	free(tasks);
	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
error:
	starpu_shutdown();
	free(tasks);
	return STARPU_TEST_SKIPPED;
}
