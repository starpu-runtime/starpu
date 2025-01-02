/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <sys/types.h>

static starpu_pthread_t thread[2];
static unsigned thread_is_initialized[2];

static starpu_pthread_cond_t cond;
static starpu_pthread_mutex_t mutex;

static size_t buffer_size = 4;
static void *cpu_buffer;
static void *gpu_buffer[2];

static starpu_pthread_cond_t cond_go;
static unsigned ready = 0;
static unsigned nready_gpu = 0;

static unsigned niter = 250000;

static starpu_pthread_cond_t cond_gpu;
static starpu_pthread_mutex_t mutex_gpu;
static unsigned data_is_available[2];

static cudaStream_t stream[2];

#define ASYNC	1
#define DO_TRANSFER_GPU_TO_RAM	1
#define DO_TRANSFER_RAM_TO_GPU	1

void send_data(unsigned src, unsigned dst)
{
	cudaError_t cures;

	/* Copy data from GPU to RAM */
#ifdef DO_TRANSFER_GPU_TO_RAM
#ifdef ASYNC
	cures = cudaMemcpyAsync(cpu_buffer, gpu_buffer[src], buffer_size, cudaMemcpyDeviceToHost, stream[src]);
	STARPU_ASSERT(!cures);

	cures = cudaStreamSynchronize(stream[src]);
	STARPU_ASSERT(!cures);
#else
	cures = cudaMemcpy(cpu_buffer, gpu_buffer[src], buffer_size, cudaMemcpyDeviceToHost);
	STARPU_ASSERT(!cures);

	cures = cudaDeviceSynchronize();
	STARPU_ASSERT(!cures);
#endif
#endif

	/* Tell the other GPU that data is in RAM */
	STARPU_PTHREAD_MUTEX_LOCK(&mutex_gpu);
	data_is_available[src] = 0;
	data_is_available[dst] = 1;
	STARPU_PTHREAD_COND_SIGNAL(&cond_gpu);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex_gpu);
	//fprintf(stderr, "SEND on %d\n", src);
}

void recv_data(unsigned src, unsigned dst)
{
	cudaError_t cures;

	/* Wait for the data to be in RAM */
	STARPU_PTHREAD_MUTEX_LOCK(&mutex_gpu);
	while (!data_is_available[dst])
	{
		STARPU_PTHREAD_COND_WAIT(&cond_gpu, &mutex_gpu);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex_gpu);
	//fprintf(stderr, "RECV on %d\n", dst);

	/* Upload data */
#ifdef DO_TRANSFER_RAM_TO_GPU
#ifdef ASYNC
	cures = cudaMemcpyAsync(gpu_buffer[dst], cpu_buffer, buffer_size, cudaMemcpyHostToDevice, stream[dst]);
	STARPU_ASSERT(!cures);
	cures = cudaStreamSynchronize(stream[dst]);
	STARPU_ASSERT(!cures);
#else
	cures = cudaMemcpy(gpu_buffer[dst], cpu_buffer, buffer_size, cudaMemcpyHostToDevice);
	STARPU_ASSERT(!cures);

	cures = cudaDeviceSynchronize();
	STARPU_ASSERT(!cures);
#endif
#endif
}

void *launch_gpu_thread(void *arg)
{
	unsigned *idptr = arg;
	unsigned id = *idptr;

	starpu_cuda_set_device(id);
	cudaFree(0);

	cudaMalloc(&gpu_buffer[id], buffer_size);
	cudaStreamCreate(&stream[id]);

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	thread_is_initialized[id] = 1;
	STARPU_PTHREAD_COND_SIGNAL(&cond);

	if (id == 0)
	{
		cudaError_t cures;
		cures = cudaHostAlloc(&cpu_buffer, buffer_size, cudaHostAllocPortable);
		STARPU_ASSERT(!cures);
		cudaDeviceSynchronize();
	}

	nready_gpu++;

	while (!ready)
		STARPU_PTHREAD_COND_WAIT(&cond_go, &mutex);

	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	unsigned iter;
	for (iter = 0; iter < niter; iter++)
	{
		if (id == 0)
		{
			send_data(0, 1);
			recv_data(1, 0);
		}
		else
		{
			recv_data(0, 1);
			send_data(1, 0);
		}
	}

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	nready_gpu--;
	STARPU_PTHREAD_COND_SIGNAL(&cond_go);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	return NULL;
}

int main(int argc, char **argv)
{

	STARPU_PTHREAD_MUTEX_INIT(&mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&cond, NULL);
	STARPU_PTHREAD_COND_INIT(&cond_go, NULL);

	unsigned id;
	for (id = 0; id < 2; id++)
	{
		thread_is_initialized[id] = 0;
		STARPU_PTHREAD_CREATE(&thread[0], NULL, launch_gpu_thread, &id);

		STARPU_PTHREAD_MUTEX_LOCK(&mutex);
		while (!thread_is_initialized[id])
		{
			 STARPU_PTHREAD_COND_WAIT(&cond, &mutex);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);
	}

	double start;
	double end;

	/* Start the ping pong */
	start = starpu_timing_now();

	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	ready = 1;
	STARPU_PTHREAD_COND_BROADCAST(&cond_go);
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	/* Wait for the end of the ping pong */
	STARPU_PTHREAD_MUTEX_LOCK(&mutex);
	while (nready_gpu > 0)
	{
		STARPU_PTHREAD_COND_WAIT(&cond_go, &mutex);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mutex);

	end = starpu_timing_now();

	double timing = end - start;

	fprintf(stderr, "Took %.0f ms for %u iterations\n", timing/1000, niter);
	fprintf(stderr, "Latency: %.2f us\n", timing/(2*niter));

	return EXIT_SUCCESS;
}
