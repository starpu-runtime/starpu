/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>

#include <starpu.h>
#include <common/config.h>
#include <core/workers.h>
#include <core/perfmodel/perfmodel.h>

#define SIZE	(32*1024*1024*sizeof(char))
#define NITER	128

#define MAXCPUS	32

struct cudadev_timing {
	int cpu_id;
	double timing_htod;
	double timing_dtoh;
};

static double bandwith_matrix[STARPU_MAXNODES][STARPU_MAXNODES] = {{-1.0}};
static double latency_matrix[STARPU_MAXNODES][STARPU_MAXNODES] = {{ -1.0}};
static unsigned was_benchmarked = 0;
static int ncuda = 0;

static int affinity_matrix[STARPU_MAXCUDADEVS][MAXCPUS];

/* Benchmarking the performance of the bus */

#ifdef USE_CUDA
static double cudadev_timing_htod[STARPU_MAXNODES] = {0.0};
static double cudadev_timing_dtoh[STARPU_MAXNODES] = {0.0};

static struct cudadev_timing cudadev_timing_per_cpu[STARPU_MAXNODES][MAXCPUS];

static void measure_bandwith_between_host_and_dev_on_cpu(int dev, int cpu)
{
	struct machine_config_s *config = _starpu_get_machine_config();
	_starpu_bind_thread_on_cpu(config, cpu);

	/* Initiliaze CUDA context on the device */
	cudaSetDevice(dev);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(config, cpu);

	/* hack to force the initialization */
	cudaFree(0);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(config, cpu);


	/* Allocate a buffer on the device */
	unsigned char *d_buffer;
	cudaMalloc((void **)&d_buffer, SIZE);
	assert(d_buffer);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(config, cpu);


	/* Allocate a buffer on the host */
	unsigned char *h_buffer;
	cudaHostAlloc((void **)&h_buffer, SIZE, 0); 
	assert(h_buffer);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(config, cpu);


	/* Fill them */
	memset(h_buffer, 0, SIZE);
	cudaMemset(d_buffer, 0, SIZE);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(config, cpu);


	unsigned iter;
	double timing;
	struct timeval start;
	struct timeval end;

	cudadev_timing_per_cpu[dev+1][cpu].cpu_id = cpu;

	/* Measure upload bandwith */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(d_buffer, h_buffer, SIZE, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();
	}
	gettimeofday(&end, NULL);
	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	cudadev_timing_per_cpu[dev+1][cpu].timing_htod = timing/NITER;

	/* Measure download bandwith */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(h_buffer, d_buffer, SIZE, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
	}
	gettimeofday(&end, NULL);
	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	cudadev_timing_per_cpu[dev+1][cpu].timing_dtoh = timing/NITER;

	/* Free buffers */
	cudaFreeHost(h_buffer);
	cudaFree(d_buffer);

	cudaThreadExit();

}

/* NB: we want to sort the bandwith by DECREASING order */
int compar_cudadev_timing(const void *left_cudadev_timing, const void *right_cudadev_timing)
{
	const struct cudadev_timing *left = left_cudadev_timing;
	const struct cudadev_timing *right = right_cudadev_timing;
	
	double left_dtoh = left->timing_dtoh;
	double left_htod = left->timing_htod;
	double right_dtoh = right->timing_dtoh;
	double right_htod = right->timing_htod;
	
	double bandwith_sum2_left = left_dtoh*left_dtoh + left_htod*left_htod;
	double bandwith_sum2_right = right_dtoh*right_dtoh + right_htod*right_htod;

	/* it's for a decreasing sorting */
	return (bandwith_sum2_left < bandwith_sum2_right);
}

static void measure_bandwith_between_host_and_dev(int dev, unsigned ncores)
{
	unsigned core;
	for (core = 0; core < ncores; core++)
	{
		measure_bandwith_between_host_and_dev_on_cpu(dev, core);
	}

	/* sort the results */
	qsort(cudadev_timing_per_cpu[dev+1], ncores,
			sizeof(struct cudadev_timing),
			compar_cudadev_timing);
	
#ifdef VERBOSE
	for (core = 0; core < ncores; core++)
	{
		unsigned current_core = cudadev_timing_per_cpu[dev+1][core].cpu_id;
		double bandwith_dtoh = cudadev_timing_per_cpu[dev+1][core].timing_dtoh;
		double bandwith_htod = cudadev_timing_per_cpu[dev+1][core].timing_htod;

		double bandwith_sum2 = bandwith_dtoh*bandwith_dtoh + bandwith_htod*bandwith_htod;

		fprintf(stderr, "BANDWITH GPU %d CPU %d - htod %lf - dtoh %lf - %lf\n", dev, current_core, bandwith_htod, bandwith_dtoh, sqrt(bandwith_sum2));
	}

	unsigned best_core = cudadev_timing_per_cpu[dev+1][0].cpu_id;

	fprintf(stderr, "BANDWITH GPU %d BEST CPU %d\n", dev, best_core);
#endif

	/* The results are sorted in a decreasing order, so that the best
	 * measurement is currently the first entry. */
	cudadev_timing_dtoh[dev+1] = cudadev_timing_per_cpu[dev+1][0].timing_dtoh;
	cudadev_timing_htod[dev+1] = cudadev_timing_per_cpu[dev+1][0].timing_htod;
}
#endif

static void benchmark_all_cuda_devices(void)
{
	int ret;

#ifdef VERBOSE
	fprintf(stderr, "Benchmarking the speed of the bus\n");
#endif

	/* Save the current cpu binding */
	cpu_set_t former_process_affinity;
	ret = sched_getaffinity(0, sizeof(former_process_affinity), &former_process_affinity);
	if (ret)
	{
		perror("sched_getaffinity");
		STARPU_ABORT();
	}

#ifdef USE_CUDA
	struct machine_config_s *config = _starpu_get_machine_config();
	unsigned ncores = _starpu_topology_get_nhwcore(config);

        cudaGetDeviceCount(&ncuda);
	int i;
	for (i = 0; i < ncuda; i++)
	{
		/* measure bandwith between Host and Device i */
		measure_bandwith_between_host_and_dev(i, ncores);
	}
#endif

	was_benchmarked = 1;

	/* Restore the former affinity */
	ret = sched_setaffinity(0, sizeof(former_process_affinity), &former_process_affinity);
	if (ret)
	{
		perror("sched_setaffinity");
		STARPU_ABORT();
	}

#ifdef VERBOSE
	fprintf(stderr, "Benchmarking the speed of the bus is done.\n");
#endif
}

static void get_bus_path(const char *type, char *path, size_t maxlen)
{
	_starpu_get_perf_model_dir_bus(path, maxlen);
	strncat(path, type, maxlen);
	
	char hostname[32];
	gethostname(hostname, 32);
	strncat(path, ".", maxlen);
	strncat(path, hostname, maxlen);
}

/*
 *	Affinity
 */

static void get_affinity_path(char *path, size_t maxlen)
{
	get_bus_path("affinity", path, maxlen);
}

static void load_bus_affinity_file_content(void)
{
	FILE *f;

	char path[256];
	get_affinity_path(path, 256);

	f = fopen(path, "r");
	STARPU_ASSERT(f);

#ifdef USE_CUDA
	struct machine_config_s *config = _starpu_get_machine_config();
	unsigned ncores = _starpu_topology_get_nhwcore(config);

        cudaGetDeviceCount(&ncuda);

	int gpu;
	for (gpu = 0; gpu < ncuda; gpu++)
	{
		int ret;

		int dummy;

		ret = fscanf(f, "%d\t", &dummy);
		STARPU_ASSERT(ret == 1);

		STARPU_ASSERT(dummy == gpu);

		unsigned core;
		for (core = 0; core < ncores; core++)
		{
			ret = fscanf(f, "%d\t", &affinity_matrix[gpu][core]);
			STARPU_ASSERT(ret == 1);
		}

		ret = fscanf(f, "\n");
		STARPU_ASSERT(ret == 0);
	}
#endif

	fclose(f);
}

static void write_bus_affinity_file_content(void)
{
	FILE *f;

	STARPU_ASSERT(was_benchmarked);

	char path[256];
	get_affinity_path(path, 256);

	f = fopen(path, "w+");
	if (!f)
	{
		perror("fopen");
		STARPU_ABORT();
	}

#ifdef USE_CUDA
	struct machine_config_s *config = _starpu_get_machine_config();
	unsigned ncores = _starpu_topology_get_nhwcore(config);

	int gpu;
	for (gpu = 0; gpu < ncuda; gpu++)
	{
		fprintf(f, "%d\t", gpu);

		unsigned core;
		for (core = 0; core < ncores; core++)
		{
			fprintf(f, "%d\t", cudadev_timing_per_cpu[gpu+1][core].cpu_id);
		}

		fprintf(f, "\n");
	}
#endif

	fclose(f);
}

static void generate_bus_affinity_file(void)
{
	if (!was_benchmarked)
		benchmark_all_cuda_devices();

	write_bus_affinity_file_content();
}

static void load_bus_affinity_file(void)
{
	int res;

	char path[256];
	get_affinity_path(path, 256);

	res = access(path, F_OK);
	if (res)
	{
		/* File does not exist yet */
		generate_bus_affinity_file();
	}

	load_bus_affinity_file_content();
}

int *get_gpu_affinity_vector(unsigned gpuid)
{
	return affinity_matrix[gpuid];
}

/*
 *	Latency
 */

static void get_latency_path(char *path, size_t maxlen)
{
	get_bus_path("latency", path, maxlen);
}

static void load_bus_latency_file_content(void)
{
	int n;
	unsigned src, dst;
	FILE *f;

	char path[256];
	get_latency_path(path, 256);

	f = fopen(path, "r");
	STARPU_ASSERT(f);

	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
		{
			double latency;

			n = fscanf(f, "%lf ", &latency);
			STARPU_ASSERT(n == 1);

			latency_matrix[src][dst] = latency;
		}

		n = fscanf(f, "\n");
		STARPU_ASSERT(n == 0);
	}

	fclose(f);
}

static void write_bus_latency_file_content(void)
{
	int src, dst;
	FILE *f;

	STARPU_ASSERT(was_benchmarked);

	char path[256];
	get_latency_path(path, 256);

	f = fopen(path, "w+");
	if (!f)
	{
		perror("fopen");
		STARPU_ABORT();
	}

	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
		{
			double latency;

			if ((src > ncuda) || (dst > ncuda))
			{
				/* convention */
				latency = -1.0;
			}
			else if (src == dst)
			{
				latency = 0.0;
			}
			else {
                                latency = ((src && dst)?2000.0:500.0);
			}

			fprintf(f, "%lf ", latency);
		}

		fprintf(f, "\n");
	}

	fclose(f);
}

static void generate_bus_latency_file(void)
{
	if (!was_benchmarked)
		benchmark_all_cuda_devices();

	write_bus_latency_file_content();
}

static void load_bus_latency_file(void)
{
	int res;

	char path[256];
	get_latency_path(path, 256);

	res = access(path, F_OK);
	if (res)
	{
		/* File does not exist yet */
		generate_bus_latency_file();
	}

	load_bus_latency_file_content();
}


/* 
 *	Bandwith
 */
static void get_bandwith_path(char *path, size_t maxlen)
{
	get_bus_path("bandwith", path, maxlen);
}

static void load_bus_bandwith_file_content(void)
{
	int n;
	unsigned src, dst;
	FILE *f;

	char path[256];
	get_bandwith_path(path, 256);

	f = fopen(path, "r");
	if (!f)
	{
		perror("fopen");
		STARPU_ABORT();
	}

	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
		{
			double bandwith;

			n = fscanf(f, "%lf ", &bandwith);
			STARPU_ASSERT(n == 1);

			bandwith_matrix[src][dst] = bandwith;
		}

		n = fscanf(f, "\n");
		STARPU_ASSERT(n == 0);
	}

	fclose(f);
}

static void write_bus_bandwith_file_content(void)
{
	int src, dst;
	FILE *f;

	STARPU_ASSERT(was_benchmarked);

	char path[256];
	get_bandwith_path(path, 256);

	f = fopen(path, "w+");
	STARPU_ASSERT(f);

	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
		{
			double bandwith;
			
			if ((src > ncuda) || (dst > ncuda))
			{
				bandwith = -1.0;
			}
#ifdef USE_CUDA
			else if (src != dst)
			{
			/* Bandwith = (SIZE)/(time i -> ram + time ram -> j)*/
				double time_src_to_ram = (src==0)?0.0:cudadev_timing_dtoh[src];
				double time_ram_to_dst = (dst==0)?0.0:cudadev_timing_htod[dst];
				
				double timing =time_src_to_ram + time_ram_to_dst;
				
				bandwith = 1.0*SIZE/timing;
			}
#endif
			else {
			        /* convention */
			        bandwith = 0.0;
			}
			
			fprintf(f, "%lf ", bandwith);
		}

		fprintf(f, "\n");
	}

	fclose(f);
}

static void generate_bus_bandwith_file(void)
{
	if (!was_benchmarked)
		benchmark_all_cuda_devices();

	write_bus_bandwith_file_content();
}

static void load_bus_bandwith_file(void)
{
	int res;

	char path[256];
	get_bandwith_path(path, 256);

	res = access(path, F_OK);
	if (res)
	{
		/* File does not exist yet */
		generate_bus_bandwith_file();
	}

	load_bus_bandwith_file_content();
}

/*
 *	Generic
 */

void starpu_force_bus_sampling(void)
{
	create_sampling_directory_if_needed();

	generate_bus_affinity_file();
	generate_bus_latency_file();
	generate_bus_bandwith_file();
}

void load_bus_performance_files(void)
{
	create_sampling_directory_if_needed();

	load_bus_affinity_file();
	load_bus_latency_file();
	load_bus_bandwith_file();
}

double predict_transfer_time(unsigned src_node, unsigned dst_node, size_t size)
{
	double bandwith = bandwith_matrix[src_node][dst_node];
	double latency = latency_matrix[src_node][dst_node];

	return latency + size/bandwith;
}
