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

#include <starpu.h>
#include <common/config.h>
#include <core/perfmodel/perfmodel.h>
#include <datawizard/data_parameters.h>

#define SIZE	(32*1024*1024*sizeof(char))
#define NITER	128

static double bandwith_matrix[MAXNODES][MAXNODES] = {{-1.0}};
static double latency_matrix[MAXNODES][MAXNODES] = {{ -1.0}};
static unsigned was_benchmarked = 0;
static int ncuda = 0;

/* Benchmarking the performance of the bus */

#ifdef USE_CUDA
static double cudadev_timing_htod[MAXNODES] = {0.0};
static double cudadev_timing_dtoh[MAXNODES] = {0.0};

static void measure_bandwith_between_host_and_dev(int dev)
{
	/* Initiliaze CUDA context on the device */
	cudaSetDevice(dev);

	/* hack to force the initialization */
	cudaFree(0);

	/* Allocate a buffer on the device */
	unsigned char *d_buffer;
	cudaMalloc((void **)&d_buffer, SIZE);
	assert(d_buffer);

	/* Allocate a buffer on the host */
	unsigned char *h_buffer;
	cudaHostAlloc((void **)&h_buffer, SIZE, 0); 
	assert(h_buffer);

	/* Fill them */
	memset(h_buffer, 0, SIZE);
	cudaMemset(d_buffer, 0, SIZE);

	unsigned iter;
	double timing;
	struct timeval start;
	struct timeval end;

	/* Measure upload bandwith */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(d_buffer, h_buffer, SIZE, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();
	}
	gettimeofday(&end, NULL);
	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	cudadev_timing_htod[dev+1] = timing/NITER;

	/* Measure download bandwith */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(h_buffer, d_buffer, SIZE, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
	}
	gettimeofday(&end, NULL);
	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	cudadev_timing_dtoh[dev+1] = timing/NITER;

	/* Free buffers */
	cudaFreeHost(h_buffer);
	cudaFree(d_buffer);

	cudaThreadExit();
}
#endif

static void benchmark_all_cuda_devices(void)
{
#ifdef VERBOSE
	fprintf(stderr, "Benchmarking the speed of the bus\n");
#endif

#ifdef USE_CUDA
        cudaGetDeviceCount(&ncuda);
	int i;
	for (i = 0; i < ncuda; i++)
	{
		/* measure bandwith between Host and Device i */
		measure_bandwith_between_host_and_dev(i);
	}
#endif

	was_benchmarked = 1;

#ifdef VERBOSE
	fprintf(stderr, "Benchmarking the speed of the bus is done.\n");
#endif
}

static void get_bus_path(const char *type, char *path, size_t maxlen)
{
	strncpy(path, PERF_MODEL_DIR_BUS, maxlen);
	strncat(path, type, maxlen);
	
	char hostname[32];
	gethostname(hostname, 32);
	strncat(path, ".", maxlen);
	strncat(path, hostname, maxlen);
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

	for (src = 0; src < MAXNODES; src++)
	{
		for (dst = 0; dst < MAXNODES; dst++)
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
	STARPU_ASSERT(f);

	for (src = 0; src < MAXNODES; src++)
	{
		for (dst = 0; dst < MAXNODES; dst++)
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

static void load_bus_latency_file(void)
{
	int res;

	char path[256];
	get_latency_path(path, 256);

	res = access(path, F_OK);
	if (res)
	{
		/* File does not exist yet */
		if (!was_benchmarked)
			benchmark_all_cuda_devices();

		write_bus_latency_file_content();
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
	STARPU_ASSERT(f);

	for (src = 0; src < MAXNODES; src++)
	{
		for (dst = 0; dst < MAXNODES; dst++)
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

	for (src = 0; src < MAXNODES; src++)
	{
		for (dst = 0; dst < MAXNODES; dst++)
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

static void load_bus_bandwith_file(void)
{
	int res;

	char path[256];
	get_bandwith_path(path, 256);

	res = access(path, F_OK);
	if (res)
	{
		/* File does not exist yet */
		if (!was_benchmarked)
			benchmark_all_cuda_devices();

		write_bus_bandwith_file_content();
	}

	load_bus_bandwith_file_content();
}

/*
 *	Generic
 */

void load_bus_performance_files(void)
{
	load_bus_latency_file();
	load_bus_bandwith_file();
}

double predict_transfer_time(unsigned src_node, unsigned dst_node, size_t size)
{
	double bandwith = bandwith_matrix[src_node][dst_node];
	double latency = latency_matrix[src_node][dst_node];

	return latency + size/bandwith;
}
