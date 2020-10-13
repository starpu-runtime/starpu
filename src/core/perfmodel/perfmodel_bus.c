/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifdef STARPU_USE_CUDA
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <sched.h>
#endif
#include <stdlib.h>
#include <math.h>

#include <starpu.h>
#include <starpu_cuda.h>
#include <starpu_opencl.h>
#include <common/config.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <core/workers.h>
#include <core/perfmodel/perfmodel.h>
#include <core/simgrid.h>
#include <core/topology.h>
#include <common/utils.h>
#include <drivers/mpi/driver_mpi_common.h>

#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif

#if defined(HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
#include <hwloc/cuda.h>
#endif

#define SIZE	(32*1024*1024*sizeof(char))
#define NITER	32

#define PATH_LENGTH 256

#ifndef STARPU_SIMGRID
static void _starpu_bus_force_sampling(void);
#endif

/* timing is in µs per byte (i.e. slowness, inverse of bandwidth) */
struct dev_timing
{
	int numa_id;
	double timing_htod;
	double latency_htod;
	double timing_dtoh;
	double latency_dtoh;
};

/* TODO: measure latency */
static double bandwidth_matrix[STARPU_MAXNODES][STARPU_MAXNODES]; /* MB/s */
static double latency_matrix[STARPU_MAXNODES][STARPU_MAXNODES]; /* µs */
static unsigned was_benchmarked = 0;
#ifndef STARPU_SIMGRID
static unsigned ncpus = 0;
#endif
static unsigned nnumas = 0;
static unsigned ncuda = 0;
static unsigned nopencl = 0;
static unsigned nmic = 0;
static unsigned nmpi_ms = 0;

/* Benchmarking the performance of the bus */

static double numa_latency[STARPU_MAXNUMANODES][STARPU_MAXNUMANODES];
static double numa_timing[STARPU_MAXNUMANODES][STARPU_MAXNUMANODES];

#ifndef STARPU_SIMGRID
static uint64_t cuda_size[STARPU_MAXCUDADEVS];
#endif
#ifdef STARPU_USE_CUDA
/* preference order of cores (logical indexes) */
static unsigned cuda_affinity_matrix[STARPU_MAXCUDADEVS][STARPU_MAXNUMANODES];

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
static double cudadev_timing_dtod[STARPU_MAXNODES][STARPU_MAXNODES] = {{0.0}};
static double cudadev_latency_dtod[STARPU_MAXNODES][STARPU_MAXNODES] = {{0.0}};
#endif
#endif
static struct dev_timing cudadev_timing_per_numa[STARPU_MAXCUDADEVS*STARPU_MAXNUMANODES];
static char cudadev_direct[STARPU_MAXNODES][STARPU_MAXNODES];
#endif

#ifndef STARPU_SIMGRID
static uint64_t opencl_size[STARPU_MAXCUDADEVS];
#endif
#ifdef STARPU_USE_OPENCL
/* preference order of cores (logical indexes) */
static unsigned opencl_affinity_matrix[STARPU_MAXOPENCLDEVS][STARPU_MAXNUMANODES];
static struct dev_timing opencldev_timing_per_numa[STARPU_MAXOPENCLDEVS*STARPU_MAXNUMANODES];
#endif

#ifdef STARPU_USE_MIC
static double mic_time_host_to_device[STARPU_MAXNODES] = {0.0};
static double mic_time_device_to_host[STARPU_MAXNODES] = {0.0};
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_MPI_MASTER_SLAVE
static double mpi_time_device_to_device[STARPU_MAXMPIDEVS][STARPU_MAXMPIDEVS] = {{0.0}};
static double mpi_latency_device_to_device[STARPU_MAXMPIDEVS][STARPU_MAXMPIDEVS] = {{0.0}};
#endif

#ifdef STARPU_HAVE_HWLOC
static hwloc_topology_t hwtopology;

hwloc_topology_t _starpu_perfmodel_get_hwtopology()
{
	return hwtopology;
}
#endif

#if (defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)) && !defined(STARPU_SIMGRID)


#ifdef STARPU_USE_CUDA

static void measure_bandwidth_between_host_and_dev_on_numa_with_cuda(int dev, int numa, int cpu, struct dev_timing *dev_timing_per_cpu)
{
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);
	size_t size = SIZE;

	/* Initialize CUDA context on the device */
	/* We do not need to enable OpenGL interoperability at this point,
	 * since we cleanly shutdown CUDA before returning. */
	cudaSetDevice(dev);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* hack to force the initialization */
	cudaFree(0);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

        /* Get the maximum size which can be allocated on the device */
	struct cudaDeviceProp prop;
	cudaError_t cures;
	cures = cudaGetDeviceProperties(&prop, dev);
	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);
	cuda_size[dev] = prop.totalGlobalMem;
        if (size > prop.totalGlobalMem/4) size = prop.totalGlobalMem/4;

	/* Allocate a buffer on the device */
	unsigned char *d_buffer;
	cures = cudaMalloc((void **)&d_buffer, size);
	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* Allocate a buffer on the host */
	unsigned char *h_buffer;

#if defined(STARPU_HAVE_HWLOC)
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	const unsigned nnuma_nodes = _starpu_topology_get_nnumanodes(config);
	if (nnuma_nodes > 1)
	{
		/* NUMA mode activated */
		hwloc_obj_t obj = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa);
#if HWLOC_API_VERSION >= 0x00020000
		h_buffer = hwloc_alloc_membind(hwtopology, size, obj->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
#else
		h_buffer = hwloc_alloc_membind_nodeset(hwtopology, size, obj->nodeset, HWLOC_MEMBIND_BIND, 0);
#endif
	}
	else
#endif
	{
		/* we use STARPU_MAIN_RAM */
		_STARPU_MALLOC(h_buffer, size);
		cudaHostRegister((void *)h_buffer, size, 0);
	}

	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* Fill them */
	memset(h_buffer, 0, size);
	cudaMemset(d_buffer, 0, size);
	cudaDeviceSynchronize();

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	const unsigned timing_numa_index = dev*STARPU_MAXNUMANODES + numa;
	unsigned iter;
	double timing;
	double start;
	double end;

	/* Measure upload bandwidth */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(d_buffer, h_buffer, size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu[timing_numa_index].timing_htod = timing/NITER/size;

	/* Measure download bandwidth */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(h_buffer, d_buffer, size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu[timing_numa_index].timing_dtoh = timing/NITER/size;

	/* Measure upload latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(d_buffer, h_buffer, 1, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu[timing_numa_index].latency_htod = timing/NITER;

	/* Measure download latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(h_buffer, d_buffer, 1, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu[timing_numa_index].latency_dtoh = timing/NITER;

	/* Free buffers */
	cudaHostUnregister(h_buffer);
#if defined(STARPU_HAVE_HWLOC)
	if (nnuma_nodes > 1)
	{
		/* NUMA mode activated */
		hwloc_free(hwtopology, h_buffer, size);
	}
	else
#endif
	{
		free(h_buffer);
	}

	cudaFree(d_buffer);

	cudaThreadExit();
}

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
static void measure_bandwidth_between_dev_and_dev_cuda(int src, int dst)
{
	size_t size = SIZE;
	int can;

	/* Get the maximum size which can be allocated on the device */
	struct cudaDeviceProp prop;
	cudaError_t cures;

	cures = cudaGetDeviceProperties(&prop, src);
	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);
	if (size > prop.totalGlobalMem/4) size = prop.totalGlobalMem/4;
	cures = cudaGetDeviceProperties(&prop, dst);
	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);
	if (size > prop.totalGlobalMem/4) size = prop.totalGlobalMem/4;

	/* Initialize CUDA context on the source */
	/* We do not need to enable OpenGL interoperability at this point,
	 * since we cleanly shutdown CUDA before returning. */
	cudaSetDevice(src);

	if (starpu_get_env_number("STARPU_ENABLE_CUDA_GPU_GPU_DIRECT") != 0)
	{
		cures = cudaDeviceCanAccessPeer(&can, src, dst);
		(void) cudaGetLastError();
		if (!cures && can)
		{
			cures = cudaDeviceEnablePeerAccess(dst, 0);
			(void) cudaGetLastError();
			if (!cures)
			{
				_STARPU_DISP("GPU-Direct %d -> %d\n", dst, src);
				cudadev_direct[src][dst] = 1;
			}
		}
	}

	/* Allocate a buffer on the device */
	unsigned char *s_buffer;
	cures = cudaMalloc((void **)&s_buffer, size);
	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);
	cudaMemset(s_buffer, 0, size);
	cudaDeviceSynchronize();

	/* Initialize CUDA context on the destination */
	/* We do not need to enable OpenGL interoperability at this point,
	 * since we cleanly shutdown CUDA before returning. */
	cudaSetDevice(dst);

	if (starpu_get_env_number("STARPU_ENABLE_CUDA_GPU_GPU_DIRECT") != 0)
	{
		cures = cudaDeviceCanAccessPeer(&can, dst, src);
		(void) cudaGetLastError();
		if (!cures && can)
		{
			cures = cudaDeviceEnablePeerAccess(src, 0);
			(void) cudaGetLastError();
			if (!cures)
			{
				_STARPU_DISP("GPU-Direct %d -> %d\n", src, dst);
				cudadev_direct[dst][src] = 1;
			}
		}
	}

	/* Allocate a buffer on the device */
	unsigned char *d_buffer;
	cures = cudaMalloc((void **)&d_buffer, size);
	if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);
	cudaMemset(d_buffer, 0, size);
	cudaDeviceSynchronize();

	unsigned iter;
	double timing;
	double start;
	double end;

	/* Measure upload bandwidth */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpyPeer(d_buffer, dst, s_buffer, src, size);
		cudaDeviceSynchronize();
	}
	end = starpu_timing_now();
	timing = end - start;

	cudadev_timing_dtod[src][dst] = timing/NITER/size;

	/* Measure upload latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpyPeer(d_buffer, dst, s_buffer, src, 1);
		cudaDeviceSynchronize();
	}
	end = starpu_timing_now();
	timing = end - start;

	cudadev_latency_dtod[src][dst] = timing/NITER;

	/* Free buffers */
	cudaFree(d_buffer);
	cudaSetDevice(src);
	cudaFree(s_buffer);

	cudaThreadExit();
}
#endif
#endif

#ifdef STARPU_USE_OPENCL
static void measure_bandwidth_between_host_and_dev_on_numa_with_opencl(int dev, int numa, int cpu, struct dev_timing *dev_timing_per_cpu)
{
	cl_context context;
	cl_command_queue queue;
	cl_int err=0;
	size_t size = SIZE;
	int not_initialized;

	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* Is the context already initialised ? */
	starpu_opencl_get_context(dev, &context);
	not_initialized = (context == NULL);
	if (not_initialized == 1)
		_starpu_opencl_init_context(dev);

	/* Get context and queue */
	starpu_opencl_get_context(dev, &context);
	starpu_opencl_get_queue(dev, &queue);

	/* Get the maximum size which can be allocated on the device */
	cl_device_id device;
	cl_ulong maxMemAllocSize, totalGlobalMem;
	starpu_opencl_get_device(dev, &device);
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
	if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
	if (size > (size_t)maxMemAllocSize/4) size = maxMemAllocSize/4;

	err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE , sizeof(totalGlobalMem), &totalGlobalMem, NULL);
	if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
	opencl_size[dev] = totalGlobalMem;

	if (_starpu_opencl_get_device_type(dev) == CL_DEVICE_TYPE_CPU)
	{
		/* Let's not use too much RAM when running OpenCL on a CPU: it
		 * would make the OS swap like crazy. */
		size /= 2;
	}

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* Allocate a buffer on the device */
	cl_mem d_buffer;
	d_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
	if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);
	/* Allocate a buffer on the host */
	unsigned char *h_buffer;
#if defined(STARPU_HAVE_HWLOC)
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	const unsigned nnuma_nodes = _starpu_topology_get_nnumanodes(config);

	if (nnuma_nodes > 1)
	{
		/* NUMA mode activated */
		hwloc_obj_t obj = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa);
#if HWLOC_API_VERSION >= 0x00020000
		h_buffer = hwloc_alloc_membind(hwtopology, size, obj->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
#else
		h_buffer = hwloc_alloc_membind_nodeset(hwtopology, size, obj->nodeset, HWLOC_MEMBIND_BIND, 0);
#endif
	}
	else
#endif
	{
		/* we use STARPU_MAIN_RAM */
		_STARPU_MALLOC(h_buffer, size);
	}

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);
	/* Fill them */
	memset(h_buffer, 0, size);
	err = clEnqueueWriteBuffer(queue, d_buffer, CL_TRUE, 0, size, h_buffer, 0, NULL, NULL);
	if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
	clFinish(queue);
	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	const unsigned timing_numa_index = dev*STARPU_MAXNUMANODES + numa;
	unsigned iter;
	double timing;
	double start;
	double end;

	/* Measure upload bandwidth */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		err = clEnqueueWriteBuffer(queue, d_buffer, CL_TRUE, 0, size, h_buffer, 0, NULL, NULL);
		if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
		clFinish(queue);
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu[timing_numa_index].timing_htod = timing/NITER/size;

	/* Measure download bandwidth */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		err = clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, size, h_buffer, 0, NULL, NULL);
		if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
		clFinish(queue);
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu[timing_numa_index].timing_dtoh = timing/NITER/size;

	/* Measure upload latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		err = clEnqueueWriteBuffer(queue, d_buffer, CL_TRUE, 0, 1, h_buffer, 0, NULL, NULL);
		if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
		clFinish(queue);
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu[timing_numa_index].latency_htod = timing/NITER;

	/* Measure download latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		err = clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, 1, h_buffer, 0, NULL, NULL);
		if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
		clFinish(queue);
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu[timing_numa_index].latency_dtoh = timing/NITER;

	/* Free buffers */
	err = clReleaseMemObject(d_buffer);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
#if defined(STARPU_HAVE_HWLOC)
	if (nnuma_nodes > 1)
	{
		/* NUMA mode activated */
		hwloc_free(hwtopology, h_buffer, size);
	}
	else
#endif
	{
		free(h_buffer);
	}

	/* Uninitiliaze OpenCL context on the device */
	if (not_initialized == 1)
		_starpu_opencl_deinit_context(dev);
}
#endif

/* NB: we want to sort the bandwidth by DECREASING order */
static int compar_dev_timing(const void *left_dev_timing, const void *right_dev_timing)
{
	const struct dev_timing *left = (const struct dev_timing *)left_dev_timing;
	const struct dev_timing *right = (const struct dev_timing *)right_dev_timing;

	double left_dtoh = left->timing_dtoh;
	double left_htod = left->timing_htod;
	double right_dtoh = right->timing_dtoh;
	double right_htod = right->timing_htod;

	double timing_sum2_left = left_dtoh*left_dtoh + left_htod*left_htod;
	double timing_sum2_right = right_dtoh*right_dtoh + right_htod*right_htod;

	/* it's for a decreasing sorting */
	return (timing_sum2_left > timing_sum2_right);
}

#ifdef STARPU_HAVE_HWLOC
static int find_cpu_from_numa_node(hwloc_obj_t obj)
{
	STARPU_ASSERT(obj);
	hwloc_obj_t current = obj;

	while (current->depth != HWLOC_OBJ_PU)
	{
		current = current->first_child;

                /* If we don't find a "PU" obj before the leave, perhaps we are
                 * just not allowed to use it. */
                if (!current)
                        return -1;
	}

	STARPU_ASSERT(current->depth == HWLOC_OBJ_PU);

	return current->logical_index;
}
#endif

static void measure_bandwidth_between_numa_nodes_and_dev(int dev, struct dev_timing *dev_timing_per_numanode, char *type)
{
	/* We measure the bandwith between each GPU and each NUMA node */
	struct _starpu_machine_config * config = _starpu_get_machine_config();
	const unsigned nnuma_nodes = _starpu_topology_get_nnumanodes(config);

	unsigned numa_id;
	for (numa_id = 0; numa_id < nnuma_nodes; numa_id++)
	{
		/* Store results by starpu id */
		const unsigned timing_numa_index = dev*STARPU_MAXNUMANODES + numa_id;

		/* Store STARPU_memnode for later */
		dev_timing_per_numanode[timing_numa_index].numa_id = numa_id;

		/* Chose one CPU connected to this NUMA node */
		int cpu_id = 0;
#ifdef STARPU_HAVE_HWLOC
		hwloc_obj_t obj = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa_id);

		if (obj)
		{
#if HWLOC_API_VERSION >= 0x00020000
			/* From hwloc 2.0, NUMAnode objects do not contain CPUs, they are contained in a group which contain the CPUs. */
			obj = obj->parent;
#endif
			cpu_id = find_cpu_from_numa_node(obj);
		}
		else
                        /* No such NUMA node, probably hwloc 1.x with no NUMA
                         * node, just take one CPU from the whole system */
			cpu_id = find_cpu_from_numa_node(hwloc_get_root_obj(hwtopology));
#endif

		if (cpu_id < 0)
			continue;

#ifdef STARPU_USE_CUDA
		if (strncmp(type, "CUDA", 4) == 0)
			measure_bandwidth_between_host_and_dev_on_numa_with_cuda(dev, numa_id, cpu_id, dev_timing_per_numanode);
#endif
#ifdef STARPU_USE_OPENCL
		if (strncmp(type, "OpenCL", 6) == 0)
			measure_bandwidth_between_host_and_dev_on_numa_with_opencl(dev, numa_id, cpu_id, dev_timing_per_numanode);
#endif
	}
}

static void measure_bandwidth_between_host_and_dev(int dev, struct dev_timing *dev_timing_per_numa, char *type)
{
	measure_bandwidth_between_numa_nodes_and_dev(dev, dev_timing_per_numa, type);

#ifdef STARPU_VERBOSE
	struct _starpu_machine_config * config = _starpu_get_machine_config();
	const unsigned nnuma_nodes = _starpu_topology_get_nnumanodes(config);
	unsigned numa_id;
	for (numa_id = 0; numa_id < nnuma_nodes; numa_id++)
	{
		const unsigned timing_numa_index = dev*STARPU_MAXNUMANODES + numa_id;
		double bandwidth_dtoh = dev_timing_per_numa[timing_numa_index].timing_dtoh;
		double bandwidth_htod = dev_timing_per_numa[timing_numa_index].timing_htod;

		double bandwidth_sum2 = bandwidth_dtoh*bandwidth_dtoh + bandwidth_htod*bandwidth_htod;

		_STARPU_DISP("(%10s) BANDWIDTH GPU %d NUMA %u - htod %f - dtoh %f - %f\n", type, dev, numa_id, bandwidth_htod, bandwidth_dtoh, sqrt(bandwidth_sum2));
	}
#endif
}
#endif /* defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) */

static void measure_bandwidth_latency_between_numa(int numa_src, int numa_dst)
{
#if defined(STARPU_HAVE_HWLOC)
	if (nnumas > 1)
	{
		/* NUMA mode activated */
		double start, end, timing;
		unsigned iter;

		unsigned char *h_buffer;
		hwloc_obj_t obj_src = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa_src);
#if HWLOC_API_VERSION >= 0x00020000
		h_buffer = hwloc_alloc_membind(hwtopology, SIZE, obj_src->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
#else
		h_buffer = hwloc_alloc_membind_nodeset(hwtopology, SIZE, obj_src->nodeset, HWLOC_MEMBIND_BIND, 0);
#endif

		unsigned char *d_buffer;
		hwloc_obj_t obj_dst = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa_dst);
#if HWLOC_API_VERSION >= 0x00020000
		d_buffer = hwloc_alloc_membind(hwtopology, SIZE, obj_dst->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
#else
		d_buffer = hwloc_alloc_membind_nodeset(hwtopology, SIZE, obj_dst->nodeset, HWLOC_MEMBIND_BIND, 0);
#endif

		memset(h_buffer, 0, SIZE);

		start = starpu_timing_now();
		for (iter = 0; iter < NITER; iter++)
		{
			memcpy(d_buffer, h_buffer, SIZE);
		}
		end = starpu_timing_now();
		timing = end - start;

		numa_timing[numa_src][numa_dst] = timing/NITER/SIZE;

		start = starpu_timing_now();
		for (iter = 0; iter < NITER; iter++)
		{
			memcpy(d_buffer, h_buffer, 1);
		}
		end = starpu_timing_now();
		timing = end - start;

		numa_latency[numa_src][numa_dst] = timing/NITER;

		hwloc_free(hwtopology, h_buffer, SIZE);
		hwloc_free(hwtopology, d_buffer, SIZE);
	}
	else
#endif
	{
		/* Cannot make a real calibration */
		numa_timing[numa_src][numa_dst] = 0.01;
		numa_latency[numa_src][numa_dst] = 0;
	}
}

static void benchmark_all_gpu_devices(void)
{
#ifdef STARPU_SIMGRID
	_STARPU_DISP("Can not measure bus in simgrid mode, please run starpu_calibrate_bus in non-simgrid mode to make sure the bus performance model was calibrated\n");
	STARPU_ABORT();
#else /* !SIMGRID */
	unsigned i, j;

	_STARPU_DEBUG("Benchmarking the speed of the bus\n");

#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_init(&hwtopology);
	_starpu_topology_filter(hwtopology);
	hwloc_topology_load(hwtopology);
#endif

#ifdef STARPU_HAVE_HWLOC
	hwloc_bitmap_t former_cpuset = hwloc_bitmap_alloc();
	hwloc_get_cpubind(hwtopology, former_cpuset, HWLOC_CPUBIND_THREAD);
#elif __linux__
	/* Save the current cpu binding */
	cpu_set_t former_process_affinity;
	int ret;
	ret = sched_getaffinity(0, sizeof(former_process_affinity), &former_process_affinity);
	if (ret)
	{
		perror("sched_getaffinity");
		STARPU_ABORT();
	}
#else
#warning Missing binding support, StarPU will not be able to properly benchmark NUMA topology
#endif

	struct _starpu_machine_config *config = _starpu_get_machine_config();
	ncpus = _starpu_topology_get_nhwcpu(config);
	nnumas = _starpu_topology_get_nnumanodes(config);

	for (i = 0; i < nnumas; i++)
		for (j = 0; j < nnumas; j++)
			if (i != j)
			{
				_STARPU_DISP("NUMA %d -> %d...\n", i, j);
				measure_bandwidth_latency_between_numa(i, j);
			}

#ifdef STARPU_USE_CUDA
	ncuda = _starpu_get_cuda_device_count();
	for (i = 0; i < ncuda; i++)
	{
		_STARPU_DISP("CUDA %u...\n", i);
		/* measure bandwidth between Host and Device i */
		measure_bandwidth_between_host_and_dev(i, cudadev_timing_per_numa, "CUDA");
	}
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	for (i = 0; i < ncuda; i++)
	{
		for (j = 0; j < ncuda; j++)
			if (i != j)
			{
				_STARPU_DISP("CUDA %u -> %u...\n", i, j);
				/* measure bandwidth between Host and Device i */
				measure_bandwidth_between_dev_and_dev_cuda(i, j);
			}
	}
#endif
#endif
#ifdef STARPU_USE_OPENCL
	nopencl = _starpu_opencl_get_device_count();
	for (i = 0; i < nopencl; i++)
	{
		_STARPU_DISP("OpenCL %u...\n", i);
		/* measure bandwith between Host and Device i */
		measure_bandwidth_between_host_and_dev(i, opencldev_timing_per_numa, "OpenCL");
	}
#endif

#ifdef STARPU_USE_MIC
	/* TODO: implement real calibration ! For now we only put an arbitrary
	 * value for each device during at the declaration as a bug fix, else
	 * we get problems on heft scheduler */
	nmic = _starpu_mic_src_get_device_count();

	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		mic_time_host_to_device[i] = 0.1;
		mic_time_device_to_host[i] = 0.1;
	}
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_MPI_MASTER_SLAVE

	_starpu_mpi_common_measure_bandwidth_latency(mpi_time_device_to_device, mpi_latency_device_to_device);

#endif /* STARPU_USE_MPI_MASTER_SLAVE */

#ifdef STARPU_HAVE_HWLOC
	hwloc_set_cpubind(hwtopology, former_cpuset, HWLOC_CPUBIND_THREAD);
	hwloc_bitmap_free(former_cpuset);
#elif __linux__
	/* Restore the former affinity */
	ret = sched_setaffinity(0, sizeof(former_process_affinity), &former_process_affinity);
	if (ret)
	{
		perror("sched_setaffinity");
		STARPU_ABORT();
	}
#endif

#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_destroy(hwtopology);
#endif

	_STARPU_DEBUG("Benchmarking the speed of the bus is done.\n");

	was_benchmarked = 1;
#endif /* !SIMGRID */
}

static void get_bus_path(const char *type, char *path, size_t maxlen)
{
	char hostname[65];

	_starpu_gethostname(hostname, sizeof(hostname));
	snprintf(path, maxlen, "%s%s.%s", _starpu_get_perf_model_dir_bus(), hostname, type);
}

/*
 *	Affinity
 */

static void get_affinity_path(char *path, size_t maxlen)
{
	get_bus_path("affinity", path, maxlen);
}

#ifndef STARPU_SIMGRID

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
static void load_bus_affinity_file_content(void)
{
	FILE *f;
	int locked;

	char path[PATH_LENGTH];
	get_affinity_path(path, sizeof(path));

	_STARPU_DEBUG("loading affinities from %s\n", path);

	f = fopen(path, "r");
	STARPU_ASSERT_MSG(f, "Error when reading from file '%s'", path);

	locked = _starpu_frdlock(f) == 0;

	unsigned gpu;

#ifdef STARPU_USE_CUDA
	ncuda = _starpu_get_cuda_device_count();
	for (gpu = 0; gpu < ncuda; gpu++)
	{
		int ret;
		unsigned dummy;

		_starpu_drop_comments(f);
		ret = fscanf(f, "%u\t", &dummy);
		STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);

		STARPU_ASSERT(dummy == gpu);

		unsigned numa;
		for (numa = 0; numa < nnumas; numa++)
		{
			ret = fscanf(f, "%u\t", &cuda_affinity_matrix[gpu][numa]);
			STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);
		}

		ret = fscanf(f, "\n");
		STARPU_ASSERT_MSG(ret == 0, "Error when reading from file '%s'", path);
	}
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	nopencl = _starpu_opencl_get_device_count();
	for (gpu = 0; gpu < nopencl; gpu++)
	{
		int ret;
		unsigned dummy;

		_starpu_drop_comments(f);
		ret = fscanf(f, "%u\t", &dummy);
		STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);

		STARPU_ASSERT(dummy == gpu);

		unsigned numa;
		for (numa = 0; numa < nnumas; numa++)
		{
			ret = fscanf(f, "%u\t", &opencl_affinity_matrix[gpu][numa]);
			STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);
		}

		ret = fscanf(f, "\n");
		STARPU_ASSERT_MSG(ret == 0, "Error when reading from file '%s'", path);
	}
#endif /* !STARPU_USE_OPENCL */
	if (locked)
		_starpu_frdunlock(f);

	fclose(f);
}
#endif /* !(STARPU_USE_CUDA_ || STARPU_USE_OPENCL */

#ifndef STARPU_SIMGRID
static void write_bus_affinity_file_content(void)
{
	STARPU_ASSERT(was_benchmarked);

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
	FILE *f;
	char path[PATH_LENGTH];
	int locked;

	get_affinity_path(path, sizeof(path));

	_STARPU_DEBUG("writing affinities to %s\n", path);

	f = fopen(path, "w+");
	if (!f)
	{
		perror("fopen write_buf_affinity_file_content");
		_STARPU_DISP("path '%s'\n", path);
		fflush(stderr);
		STARPU_ABORT();
	}

	locked = _starpu_frdlock(f) == 0;
	unsigned numa;
	unsigned gpu;

	fprintf(f, "# GPU\t");
	for (numa = 0; numa < nnumas; numa++)
		fprintf(f, "NUMA%u\t", numa);
	fprintf(f, "\n");

#ifdef STARPU_USE_CUDA
	{
		/* Use an other array to sort bandwidth */
		struct dev_timing cudadev_timing_per_numa_sorted[STARPU_MAXCUDADEVS*STARPU_MAXNUMANODES];
		memcpy(cudadev_timing_per_numa_sorted, cudadev_timing_per_numa, STARPU_MAXCUDADEVS*STARPU_MAXNUMANODES*sizeof(struct dev_timing));

		for (gpu = 0; gpu < ncuda; gpu++)
		{
			fprintf(f, "%u\t", gpu);

			qsort(&(cudadev_timing_per_numa_sorted[gpu*STARPU_MAXNUMANODES]), nnumas, sizeof(struct dev_timing), compar_dev_timing);

			for (numa = 0; numa < nnumas; numa++)
			{
				fprintf(f, "%d\t", cudadev_timing_per_numa_sorted[gpu*STARPU_MAXNUMANODES+numa].numa_id);
			}

			fprintf(f, "\n");
		}
	}
#endif
#ifdef STARPU_USE_OPENCL
	{
		/* Use an other array to sort bandwidth */
		struct dev_timing opencldev_timing_per_numa_sorted[STARPU_MAXOPENCLDEVS*STARPU_MAXNUMANODES];
		memcpy(opencldev_timing_per_numa_sorted, opencldev_timing_per_numa, STARPU_MAXOPENCLDEVS*STARPU_MAXNUMANODES*sizeof(struct dev_timing));

		for (gpu = 0; gpu < nopencl; gpu++)
		{
			fprintf(f, "%u\t", gpu);

			qsort(&(opencldev_timing_per_numa_sorted[gpu*STARPU_MAXNUMANODES]), nnumas, sizeof(struct dev_timing), compar_dev_timing);

			for (numa = 0; numa < nnumas; numa++)
			{
				fprintf(f, "%d\t", opencldev_timing_per_numa_sorted[gpu*STARPU_MAXNUMANODES+numa].numa_id);
			}

			fprintf(f, "\n");
		}
	}
#endif

	if (locked)
		_starpu_frdunlock(f);
	fclose(f);
#endif
}
#endif /* STARPU_SIMGRID */

static void generate_bus_affinity_file(void)
{
	if (!was_benchmarked)
		benchmark_all_gpu_devices();

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	/* Slaves don't write files */
	if (!_starpu_mpi_common_is_src_node())
		return;
#endif

	write_bus_affinity_file_content();
}

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
static int check_bus_affinity_file(void)
{
	int ret = 1;
	FILE *f;
	int locked;
	unsigned dummy;

	char path[PATH_LENGTH];
	get_affinity_path(path, sizeof(path));

	_STARPU_DEBUG("loading affinities from %s\n", path);

	f = fopen(path, "r");
	STARPU_ASSERT_MSG(f, "Error when reading from file '%s'", path);

	locked = _starpu_frdlock(f) == 0;

	ret = fscanf(f, "# GPU\t");
	STARPU_ASSERT_MSG(ret == 0, "Error when reading from file '%s'", path);

	ret = fscanf(f, "NUMA%u\t", &dummy);

	if (locked)
		_starpu_frdunlock(f);

	fclose(f);
	return ret == 1;
}
#endif

static void load_bus_affinity_file(void)
{
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
	int exist, check = 1;

	char path[PATH_LENGTH];
	get_affinity_path(path, sizeof(path));

	/* access return 0 if file exists */
	exist = access(path, F_OK);

	if (exist == 0)
		/* return 0 if it's not good */
		check = check_bus_affinity_file();

	if (check == 0)
		_STARPU_DISP("Affinity File is too old for this version of StarPU ! Rebuilding it...\n");

	if (check == 0 || exist != 0)
	{
		/* File does not exist yet */
		generate_bus_affinity_file();
	}

	load_bus_affinity_file_content();
#endif
}

#ifdef STARPU_USE_CUDA
unsigned *_starpu_get_cuda_affinity_vector(unsigned gpuid)
{
	return cuda_affinity_matrix[gpuid];
}
#endif /* STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
unsigned *_starpu_get_opencl_affinity_vector(unsigned gpuid)
{
	return opencl_affinity_matrix[gpuid];
}
#endif /* STARPU_USE_OPENCL */

void starpu_bus_print_affinity(FILE *f)
{
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
	unsigned numa;
	unsigned gpu;
#endif

	fprintf(f, "# GPU\tNUMA in preference order (logical index)\n");

#ifdef STARPU_USE_CUDA
	fprintf(f, "# CUDA\n");
	for(gpu = 0 ; gpu<ncuda ; gpu++)
	{
		fprintf(f, "%u\t", gpu);
		for (numa = 0; numa < nnumas; numa++)
		{
			fprintf(f, "%u\t", cuda_affinity_matrix[gpu][numa]);
		}
		fprintf(f, "\n");
	}
#endif
#ifdef STARPU_USE_OPENCL
	fprintf(f, "# OpenCL\n");
	for(gpu = 0 ; gpu<nopencl ; gpu++)
	{
		fprintf(f, "%u\t", gpu);
		for (numa = 0; numa < nnumas; numa++)
		{
			fprintf(f, "%u\t", opencl_affinity_matrix[gpu][numa]);
		}
		fprintf(f, "\n");
	}
#endif
}
#endif /* STARPU_SIMGRID */

/*
 *	Latency
 */

static void get_latency_path(char *path, size_t maxlen)
{
	get_bus_path("latency", path, maxlen);
}

static int load_bus_latency_file_content(void)
{
	int n;
	unsigned src, dst;
	FILE *f;
	double latency;
	int locked;

	char path[PATH_LENGTH];
	get_latency_path(path, sizeof(path));

	_STARPU_DEBUG("loading latencies from %s\n", path);

	f = fopen(path, "r");
	if (!f)
	{
		perror("fopen load_bus_latency_file_content");
		_STARPU_DISP("path '%s'\n", path);
		fflush(stderr);
		STARPU_ABORT();
	}
	locked = _starpu_frdlock(f) == 0;

	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		_starpu_drop_comments(f);
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
		{
			n = _starpu_read_double(f, "%le", &latency);
			if (n != 1)
			{
				_STARPU_DISP("Error while reading latency file <%s>. Expected a number. Did you change the maximum number of GPUs at ./configure time?\n", path);
				fclose(f);
				return 0;
			}
			n = getc(f);
			if (n == '\n')
				break;
			if (n != '\t')
			{
				_STARPU_DISP("bogus character '%c' (%d) in latency file %s\n", n, n, path);
				fclose(f);
				return 0;
			}

			latency_matrix[src][dst] = latency;

			/* Look out for \t\n */
			n = getc(f);
			if (n == '\n')
				break;
			ungetc(n, f);
			n = '\t';
		}

		/* No more values, take NAN */
		for ( ; dst < STARPU_MAXNODES; dst++)
			latency_matrix[src][dst] = NAN;

		while (n == '\t')
		{
			/* Look out for \t\n */
			n = getc(f);
			if (n == '\n')
				break;
			ungetc(n, f);

			n = _starpu_read_double(f, "%le", &latency);
			if (n && !isnan(latency))
			{
				_STARPU_DISP("Too many nodes in latency file %s for this configuration (%d). Did you change the maximum number of GPUs at ./configure time?\n", path, STARPU_MAXNODES);
				fclose(f);
				return 0;
			}
			n = getc(f);
		}
		if (n != '\n')
		{
			_STARPU_DISP("Bogus character '%c' (%d) in latency file %s\n", n, n, path);
			fclose(f);
			return 0;
		}

		/* Look out for EOF */
		n = getc(f);
		if (n == EOF)
			break;
		ungetc(n, f);
	}
	if (locked)
		_starpu_frdunlock(f);
	fclose(f);

	/* No more values, take NAN */
	for ( ; src < STARPU_MAXNODES; src++)
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
			latency_matrix[src][dst] = NAN;

	return 1;
}

#if !defined(STARPU_SIMGRID) && (defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL))
static double search_bus_best_latency(int src, char * type, int htod)
{
	/* Search the best latency for this node */
	double best = 0.0;
	double actual = 0.0;
	unsigned check = 0;
	unsigned numa;
	for (numa = 0; numa < nnumas; numa++)
	{
#ifdef STARPU_USE_CUDA
		if (strncmp(type, "CUDA", 4) == 0)
		{
			if (htod)
				actual = cudadev_timing_per_numa[src*STARPU_MAXNUMANODES+numa].latency_htod;
			else
				actual = cudadev_timing_per_numa[src*STARPU_MAXNUMANODES+numa].latency_dtoh;
		}
#endif
#ifdef STARPU_USE_OPENCL
		if (strncmp(type, "OpenCL", 6) == 0)
		{
			if (htod)
				actual = opencldev_timing_per_numa[src*STARPU_MAXNUMANODES+numa].latency_htod;
			else
				actual = opencldev_timing_per_numa[src*STARPU_MAXNUMANODES+numa].latency_dtoh;
		}
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
		if (!check || actual < best)
		{
			best = actual;
			check = 1;
		}
#endif
	}
	return best;
}
#endif

#if !defined(STARPU_SIMGRID) 
static void write_bus_latency_file_content(void)
{
	unsigned src, dst, maxnode;
	/* Boundaries to check if src or dst are inside the interval */
	unsigned b_low, b_up;
	FILE *f;
	int locked;

	STARPU_ASSERT(was_benchmarked);

	char path[PATH_LENGTH];
	get_latency_path(path, sizeof(path));

	_STARPU_DEBUG("writing latencies to %s\n", path);

	f = fopen(path, "w+");
	if (!f)
	{
		perror("fopen write_bus_latency_file_content");
		_STARPU_DISP("path '%s'\n", path);
		fflush(stderr);
		STARPU_ABORT();
	}
	locked = _starpu_fwrlock(f) == 0;
	_starpu_fftruncate(f, 0);

	fprintf(f, "# ");
	for (dst = 0; dst < STARPU_MAXNODES; dst++)
		fprintf(f, "to %u\t\t", dst);
	fprintf(f, "\n");

	maxnode = nnumas;
#ifdef STARPU_USE_CUDA
	maxnode += ncuda;
#endif
#ifdef STARPU_USE_OPENCL
	maxnode += nopencl;
#endif
#ifdef STARPU_USE_MIC
	maxnode += nmic;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	maxnode += nmpi_ms;
#endif
	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
		{
			/* µs */
			double latency = 0.0;

			if ((src >= maxnode) || (dst >= maxnode))
			{
				/* convention */
				latency = NAN;
			}
			else if (src == dst)
			{
				latency = 0.0;
			}
			else
			{
				b_low = b_up = 0;

				/* ---- Begin NUMA ---- */
				b_up += nnumas;

				if (src >= b_low && src < b_up && dst >= b_low && dst < b_up)
					latency += numa_latency[src-b_low][dst-b_low];

				/* copy interval to check numa index later */
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
				unsigned numa_low = b_low;
				unsigned numa_up = b_up;
#endif

				b_low += nnumas;
				/* ---- End NUMA ---- */
#ifdef STARPU_USE_CUDA
				b_up += ncuda;
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
				if (src >= b_low && src < b_up && dst >= b_low && dst < b_up)
					latency += cudadev_latency_dtod[src-b_low][dst-b_low];
				else
#endif
				{
					/* Check if it's CUDA <-> NUMA link */
					if (src >=b_low && src < b_up && dst >= numa_low && dst < numa_up)
						latency += cudadev_timing_per_numa[(src-b_low)*STARPU_MAXNUMANODES+dst-numa_low].latency_dtoh;
					if (dst >= b_low && dst < b_up && src >= numa_low && dst < numa_up)
						latency += cudadev_timing_per_numa[(dst-b_low)*STARPU_MAXNUMANODES+src-numa_low].latency_htod;
					/* To other devices, take the best latency */
					if (src >= b_low && src < b_up && !(dst >= numa_low && dst < numa_up))
						latency += search_bus_best_latency(src-b_low, "CUDA", 0);
					if (dst >= b_low && dst < b_up && !(src >= numa_low && dst < numa_up))
						latency += search_bus_best_latency(dst-b_low, "CUDA", 1);
				}
				b_low += ncuda;
#endif

#ifdef STARPU_USE_OPENCL
				b_up += nopencl;
				/* Check if it's OpenCL <-> NUMA link */
				if (src >= b_low && src < b_up && dst >= numa_low && dst < numa_up)
					latency += opencldev_timing_per_numa[(src-b_low)*STARPU_MAXNUMANODES+dst-numa_low].latency_dtoh;
				if (dst >= b_low && dst < b_up && src >= numa_low && dst < numa_up)
					latency += opencldev_timing_per_numa[(dst-b_low)*STARPU_MAXNUMANODES+src-numa_low].latency_htod;
				/* To other devices, take the best latency */
				if (src >= b_low && src < b_up && !(dst >= numa_low && dst < numa_up))
						latency += search_bus_best_latency(src-b_low, "OpenCL", 0);
				if (dst >= b_low && dst < b_up && !(src >= numa_low && dst < numa_up))
						latency += search_bus_best_latency(dst-b_low, "OpenCL", 1);
				b_low += nopencl;
#endif
#ifdef STARPU_USE_MIC
				b_up += nmic;
				/* TODO Latency MIC */
				b_low += nmic;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
				b_up += nmpi_ms;
				/* Modify MPI src and MPI dst if they contain the master node or not
				 * Because, we only take care about slaves */
				int mpi_master = _starpu_mpi_common_get_src_node();

				int mpi_src = src - b_low;
				mpi_src = (mpi_master <= mpi_src) ? mpi_src+1 : mpi_src;

				int mpi_dst = dst - b_low;
				mpi_dst = (mpi_master <= mpi_dst) ? mpi_dst+1 : mpi_dst;

				if (src >= b_low && src < b_up && dst >= b_low && dst < b_up)
					latency += mpi_latency_device_to_device[mpi_src][mpi_dst];
				else
				{
					if (src >= b_low && src < b_up)
						latency += mpi_latency_device_to_device[mpi_src][mpi_master];
					if (dst >= b_low && dst < b_up)
						latency += mpi_latency_device_to_device[mpi_master][mpi_dst];
				}
				b_low += nmpi_ms;
#endif
			}

			if (dst > 0)
				fputc('\t', f);
			_starpu_write_double(f, "%e", latency);
		}

		fprintf(f, "\n");
	}
	if (locked)
		_starpu_fwrunlock(f);

	fclose(f);
}
#endif

static void generate_bus_latency_file(void)
{
	if (!was_benchmarked)
		benchmark_all_gpu_devices();

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	/* Slaves don't write files */
	if (!_starpu_mpi_common_is_src_node())
		return;
#endif

#ifndef STARPU_SIMGRID
	write_bus_latency_file_content();
#endif
}

static void load_bus_latency_file(void)
{
	int res;

	char path[PATH_LENGTH];
	get_latency_path(path, sizeof(path));

	res = access(path, F_OK);
	if (res || !load_bus_latency_file_content())
	{
		/* File does not exist yet or is bogus */
		generate_bus_latency_file();
		res = load_bus_latency_file_content();
		STARPU_ASSERT(res);
	}

}


/*
 *	Bandwidth
 */
static void get_bandwidth_path(char *path, size_t maxlen)
{
	get_bus_path("bandwidth", path, maxlen);
}

static int load_bus_bandwidth_file_content(void)
{
	int n;
	unsigned src, dst;
	FILE *f;
	double bandwidth;
	int locked;

	char path[PATH_LENGTH];
	get_bandwidth_path(path, sizeof(path));

	_STARPU_DEBUG("loading bandwidth from %s\n", path);

	f = fopen(path, "r");
	if (!f)
	{
		perror("fopen load_bus_bandwidth_file_content");
		_STARPU_DISP("path '%s'\n", path);
		fflush(stderr);
		STARPU_ABORT();
	}
	locked = _starpu_frdlock(f) == 0;

	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		_starpu_drop_comments(f);
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
		{
			n = _starpu_read_double(f, "%le", &bandwidth);
			if (n != 1)
			{
				_STARPU_DISP("Error while reading bandwidth file <%s>. Expected a number\n", path);
				fclose(f);
				return 0;
			}
			n = getc(f);
			if (n == '\n')
				break;
			if (n != '\t')
			{
				_STARPU_DISP("bogus character '%c' (%d) in bandwidth file %s\n", n, n, path);
				fclose(f);
				return 0;
			}

			int limit_bandwidth = starpu_get_env_number("STARPU_LIMIT_BANDWIDTH");
			if (limit_bandwidth >= 0)
			{
#ifndef STARPU_SIMGRID
				_STARPU_DISP("Warning: STARPU_LIMIT_BANDWIDTH set to %d but simgrid not enabled, thus ignored\n", limit_bandwidth);
#else
#ifdef HAVE_SG_LINK_BANDWIDTH_SET
				bandwidth = limit_bandwidth;
#else
				_STARPU_DISP("Warning: STARPU_LIMIT_BANDWIDTH set to %d but this requires simgrid 3.26\n", limit_bandwidth);
#endif
#endif
			}

			bandwidth_matrix[src][dst] = bandwidth;

			/* Look out for \t\n */
			n = getc(f);
			if (n == '\n')
				break;
			ungetc(n, f);
			n = '\t';
		}

		/* No more values, take NAN */
		for ( ; dst < STARPU_MAXNODES; dst++)
			bandwidth_matrix[src][dst] = NAN;

		while (n == '\t')
		{
			/* Look out for \t\n */
			n = getc(f);
			if (n == '\n')
				break;
			ungetc(n, f);

			n = _starpu_read_double(f, "%le", &bandwidth);
			if (n && !isnan(bandwidth))
			{
				_STARPU_DISP("Too many nodes in bandwidth file %s for this configuration (%d)\n", path, STARPU_MAXNODES);
				fclose(f);
				return 0;
			}
			n = getc(f);
		}
		if (n != '\n')
		{
			_STARPU_DISP("Bogus character '%c' (%d) in bandwidth file %s\n", n, n, path);
			fclose(f);
			return 0;
		}

		/* Look out for EOF */
		n = getc(f);
		if (n == EOF)
			break;
		ungetc(n, f);
	}
	if (locked)
		_starpu_frdunlock(f);
	fclose(f);

	/* No more values, take NAN */
	for ( ; src < STARPU_MAXNODES; src++)
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
			latency_matrix[src][dst] = NAN;

	return 1;
}

#if !defined(STARPU_SIMGRID) && (defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL))
static double search_bus_best_timing(int src, char * type, int htod)
{
        /* Search the best latency for this node */
        double best = 0.0;
        double actual = 0.0;
        unsigned check = 0;
        unsigned numa;
        for (numa = 0; numa < nnumas; numa++)
        {
#ifdef STARPU_USE_CUDA
                if (strncmp(type, "CUDA", 4) == 0)
		{
                        if (htod)
                                actual = cudadev_timing_per_numa[src*STARPU_MAXNUMANODES+numa].timing_htod;
                        else
                                actual = cudadev_timing_per_numa[src*STARPU_MAXNUMANODES+numa].timing_dtoh;
		}
#endif
#ifdef STARPU_USE_OPENCL
		if (strncmp(type, "OpenCL", 6) == 0)
		{
			if (htod)
				actual = opencldev_timing_per_numa[src*STARPU_MAXNUMANODES+numa].timing_htod;
			else
				actual = opencldev_timing_per_numa[src*STARPU_MAXNUMANODES+numa].timing_dtoh;
		}
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
                if (!check || actual < best)
                {
                        best = actual;
                        check = 1;
                }
#endif
        }
        return best;
}
#endif

#if !defined(STARPU_SIMGRID)
static void write_bus_bandwidth_file_content(void)
{
	unsigned src, dst, maxnode;
	unsigned b_low, b_up;
	FILE *f;
	int locked;

	STARPU_ASSERT(was_benchmarked);

	char path[PATH_LENGTH];
	get_bandwidth_path(path, sizeof(path));

	_STARPU_DEBUG("writing bandwidth to %s\n", path);

	f = fopen(path, "w+");
	STARPU_ASSERT_MSG(f, "Error when opening file (writing) '%s'", path);

	locked = _starpu_fwrlock(f) == 0;
	_starpu_fftruncate(f, 0);

	fprintf(f, "# ");
	for (dst = 0; dst < STARPU_MAXNODES; dst++)
		fprintf(f, "to %u\t\t", dst);
	fprintf(f, "\n");

	maxnode = nnumas;
#ifdef STARPU_USE_CUDA
	maxnode += ncuda;
#endif
#ifdef STARPU_USE_OPENCL
	maxnode += nopencl;
#endif
#ifdef STARPU_USE_MIC
	maxnode += nmic;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	maxnode += nmpi_ms;
#endif
	for (src = 0; src < STARPU_MAXNODES; src++)
	{
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
		{
			double bandwidth;

			if ((src >= maxnode) || (dst >= maxnode))
			{
				bandwidth = NAN;
			}
			else if (src != dst)
			{
				double slowness = 0.0;
				/* Total bandwidth is the harmonic mean of bandwidths */
				b_low = b_up = 0;

				/* Begin NUMA */
				b_up += nnumas;

				if (src >= b_low && src < b_up && dst >= b_low && dst < b_up)
                                        slowness += numa_timing[src-b_low][dst-b_low];

                                /* copy interval to check numa index later */
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
                                unsigned numa_low = b_low;
                                unsigned numa_up = b_up;
#endif

				b_low += nnumas;
				/* End NUMA */
#ifdef STARPU_USE_CUDA
				b_up += ncuda;
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
				if (src >= b_low && src < b_up && dst >= b_low && dst < b_up)
					/* Direct GPU-GPU transfert */
					slowness += cudadev_timing_dtod[src-b_low][dst-b_low];
				else
#endif
				{
                                        /* Check if it's CUDA <-> NUMA link */
                                        if (src >= b_low && src < b_up && dst >= numa_low && dst < numa_up)
                                                slowness += cudadev_timing_per_numa[(src-b_low)*STARPU_MAXNUMANODES+dst-numa_low].timing_dtoh;
                                        if (dst >= b_low && dst < b_up && src >= numa_low && dst < numa_up)
                                                slowness += cudadev_timing_per_numa[(dst-b_low)*STARPU_MAXNUMANODES+src-numa_low].timing_htod;
                                        /* To other devices, take the best slowness */
                                        if (src >= b_low && src < b_up && !(dst >= numa_low && dst < numa_up))
                                                slowness += search_bus_best_timing(src-b_low, "CUDA", 0);
                                        if (dst >= b_low && dst < b_up && !(src >= numa_low && dst < numa_up))
                                                slowness += search_bus_best_timing(dst-b_low, "CUDA", 1);
				}
				b_low += ncuda;
#endif
#ifdef STARPU_USE_OPENCL
				b_up += nopencl;
				  /* Check if it's OpenCL <-> NUMA link */
                                if (src >= b_low && src < b_up && dst >= numa_low && dst < numa_up)
                                        slowness += opencldev_timing_per_numa[(src-b_low)*STARPU_MAXNUMANODES+dst-numa_low].timing_dtoh;
                                if (dst >= b_low && dst < b_up && src >= numa_low && dst < numa_up)
                                        slowness += opencldev_timing_per_numa[(dst-b_low)*STARPU_MAXNUMANODES+src-numa_low].timing_htod;
                                /* To other devices, take the best slowness */
                                if (src >= b_low && src < b_up && !(dst >= numa_low && dst < numa_up))
					slowness += search_bus_best_timing(src-b_low, "OpenCL", 0);
				if (dst >= b_low && dst < b_up && !(src >= numa_low && dst < numa_up))
					slowness += search_bus_best_timing(dst-b_low, "OpenCL", 1);
				b_low += nopencl;
#endif
#ifdef STARPU_USE_MIC
				b_up += nmic;
				if (src >= b_low && src < b_up)
					slowness += mic_time_device_to_host[src-b_low];
				if (dst >= b_low && dst < b_up)
					slowness += mic_time_host_to_device[dst-b_low];
				b_low += nmic;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
				b_up += nmpi_ms;
				/* Modify MPI src and MPI dst if they contain the master node or not
				 * Because, we only take care about slaves */
				int mpi_master = _starpu_mpi_common_get_src_node();

				int mpi_src = src - b_low;
				mpi_src = (mpi_master <= mpi_src) ? mpi_src+1 : mpi_src;

				int mpi_dst = dst - b_low;
				mpi_dst = (mpi_master <= mpi_dst) ? mpi_dst+1 : mpi_dst;

                                if (src >= b_low && src < b_up && dst >= b_low && dst < b_up)
                                        slowness += mpi_time_device_to_device[mpi_src][mpi_dst];
                                else
                                {
                                        if (src >= b_low && src < b_up)
                                                slowness += mpi_time_device_to_device[mpi_src][mpi_master];
                                        if (dst >= b_low && dst < b_up)
                                                slowness += mpi_time_device_to_device[mpi_master][mpi_dst];
                                }

				b_low += nmpi_ms;
#endif
				bandwidth = 1.0/slowness;
			}
			else
			{
				/* convention */
				bandwidth = 0.0;
			}

			if (dst)
				fputc('\t', f);
			_starpu_write_double(f, "%e", bandwidth);
		}

		fprintf(f, "\n");
	}

	if (locked)
		_starpu_fwrunlock(f);
	fclose(f);
}
#endif /* STARPU_SIMGRID */

void starpu_bus_print_filenames(FILE *output)
{
	char bandwidth_path[PATH_LENGTH];
	char affinity_path[PATH_LENGTH];
	char latency_path[PATH_LENGTH];

	get_bandwidth_path(bandwidth_path, sizeof(bandwidth_path));
	get_affinity_path(affinity_path, sizeof(affinity_path));
	get_latency_path(latency_path, sizeof(latency_path));

	fprintf(output, "bandwidth: <%s>\n", bandwidth_path);
	fprintf(output, " affinity: <%s>\n", affinity_path);
	fprintf(output, "  latency: <%s>\n", latency_path);
}

void starpu_bus_print_bandwidth(FILE *f)
{
	unsigned src, dst, maxnode = starpu_memory_nodes_get_count();

	fprintf(f, "from/to\t");
	for (dst = 0; dst < maxnode; dst++)
	{
		char name[128];
		starpu_memory_node_get_name(dst, name, sizeof(name));
		fprintf(f, "%s\t", name);
	}
	fprintf(f, "\n");

	for (src = 0; src < maxnode; src++)
	{
		char name[128];
		starpu_memory_node_get_name(src, name, sizeof(name));
		fprintf(f, "%s\t", name);

		for (dst = 0; dst < maxnode; dst++)
			fprintf(f, "%.0f\t", bandwidth_matrix[src][dst]);

		fprintf(f, "\n");
	}
	fprintf(f, "\n");

	for (src = 0; src < maxnode; src++)
	{
		char name[128];
		starpu_memory_node_get_name(src, name, sizeof(name));
		fprintf(f, "%s\t", name);

		for (dst = 0; dst < maxnode; dst++)
			fprintf(f, "%.0f\t", latency_matrix[src][dst]);

		fprintf(f, "\n");
	}

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
	if (ncuda != 0 || nopencl != 0)
		fprintf(f, "\nGPU\tNUMA in preference order (logical index), host-to-device, device-to-host\n");
	for (src = 0; src < ncuda + nopencl; src++)
	{
		struct dev_timing *timing;
		struct _starpu_machine_config * config = _starpu_get_machine_config();
		unsigned config_nnumas = _starpu_topology_get_nnumanodes(config);
		unsigned numa;

#ifdef STARPU_USE_CUDA
		if (src < ncuda)
		{
			fprintf(f, "CUDA_%u\t", src);
			for (numa = 0; numa < config_nnumas; numa++)
			{
				timing = &cudadev_timing_per_numa[src*STARPU_MAXNUMANODES+numa];
				if (timing->timing_htod)
					fprintf(f, "%2d %.0f %.0f\t", timing->numa_id, 1/timing->timing_htod, 1/timing->timing_dtoh);
				else
					fprintf(f, "%2u\t", cuda_affinity_matrix[src][numa]);
			}
		}
#ifdef STARPU_USE_OPENCL
		else
#endif
#endif
#ifdef STARPU_USE_OPENCL
		{
			fprintf(f, "OpenCL%u\t", src-ncuda);
			for (numa = 0; numa < config_nnumas; numa++)
			{
				timing = &opencldev_timing_per_numa[(src-ncuda)*STARPU_MAXNUMANODES+numa];
				if (timing->timing_htod)
					fprintf(f, "%2d %.0f %.0f\t", timing->numa_id, 1/timing->timing_htod, 1/timing->timing_dtoh);
				else
					fprintf(f, "%2u\t", opencl_affinity_matrix[src][numa]);
			}
		}
#endif
		fprintf(f, "\n");
	}
#endif
}

static void generate_bus_bandwidth_file(void)
{
	if (!was_benchmarked)
		benchmark_all_gpu_devices();

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	/* Slaves don't write files */
	if (!_starpu_mpi_common_is_src_node())
		return;
#endif

#ifndef STARPU_SIMGRID
	write_bus_bandwidth_file_content();
#endif
}

static void load_bus_bandwidth_file(void)
{
	int res;

	char path[PATH_LENGTH];
	get_bandwidth_path(path, sizeof(path));

	res = access(path, F_OK);
	if (res || !load_bus_bandwidth_file_content())
	{
		/* File does not exist yet or is bogus */
		generate_bus_bandwidth_file();
		res = load_bus_bandwidth_file_content();
		STARPU_ASSERT(res);
	}
}

#ifndef STARPU_SIMGRID
/*
 *	Config
 */
static void get_config_path(char *path, size_t maxlen)
{
	get_bus_path("config", path, maxlen);
}

#if defined(STARPU_USE_MPI_MASTER_SLAVE)
/* check if the master or one slave has to recalibrate */
static int mpi_check_recalibrate(int my_recalibrate)
{
	int nb_mpi = _starpu_mpi_src_get_device_count() + 1;
	int mpi_recalibrate[nb_mpi];
	int i;

	MPI_Allgather(&my_recalibrate, 1, MPI_INT, mpi_recalibrate, 1, MPI_INT, MPI_COMM_WORLD);

        for (i = 0; i < nb_mpi; i++)
        {
                if (mpi_recalibrate[i])
                {
                        return 1;
                }
        }
        return 0;
}
#endif

static void compare_value_and_recalibrate(char * msg, unsigned val_file, unsigned val_detected)
{
	int recalibrate = 0;
	if (val_file != val_detected)
		recalibrate = 1;

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	//Send to each other to know if we had to recalibrate because someone cannot have the correct value in the config file
	recalibrate = mpi_check_recalibrate(recalibrate);
#endif

	if (recalibrate)
	{
#ifdef STARPU_USE_MPI_MASTER_SLAVE
		/* Only the master prints the message */
		if (_starpu_mpi_common_is_src_node())
#endif
			_STARPU_DISP("Current configuration does not match the bus performance model (%s: (stored) %d != (current) %d), recalibrating...\n", msg, val_file, val_detected);

		_starpu_bus_force_sampling();

#ifdef STARPU_USE_MPI_MASTER_SLAVE
		if (_starpu_mpi_common_is_src_node())
#endif
			_STARPU_DISP("... done\n");
	}
}

static void check_bus_config_file(void)
{
	int res;
	char path[PATH_LENGTH];
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	int recalibrate = 0;

	get_config_path(path, sizeof(path));
	res = access(path, F_OK);

	if (res || config->conf.bus_calibrate > 0)
		recalibrate = 1;

#if defined(STARPU_USE_MPI_MASTER_SLAVE)
	//Send to each other to know if we had to recalibrate because someone cannot have the config file
	recalibrate = mpi_check_recalibrate(recalibrate);
#endif

	if (recalibrate)
	{
		if (res)
			_STARPU_DISP("No performance model for the bus, calibrating...\n");
		_starpu_bus_force_sampling();
		if (res)
			_STARPU_DISP("... done\n");
	}
	else
	{
		FILE *f;
		int ret;
		unsigned read_cuda = -1, read_opencl = -1, read_mic = -1, read_mpi_ms = -1;
		unsigned read_cpus = -1, read_numa = -1;
		int locked;

		// Loading configuration from file
		f = fopen(path, "r");
		STARPU_ASSERT_MSG(f, "Error when reading from file '%s'", path);
		locked = _starpu_frdlock(f) == 0;
		_starpu_drop_comments(f);

		ret = fscanf(f, "%u\t", &read_cpus);
		STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);
		_starpu_drop_comments(f);

		ret = fscanf(f, "%u\t", &read_numa);
		STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);
		_starpu_drop_comments(f);

		ret = fscanf(f, "%u\t", &read_cuda);
		STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);
		_starpu_drop_comments(f);

		ret = fscanf(f, "%u\t", &read_opencl);
		STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);
		_starpu_drop_comments(f);

		ret = fscanf(f, "%u\t", &read_mic);
		if (ret == 0)
			read_mic = 0;
		_starpu_drop_comments(f);

		ret = fscanf(f, "%u\t", &read_mpi_ms);
		if (ret == 0)
			read_mpi_ms = 0;
		_starpu_drop_comments(f);

		if (locked)
			_starpu_frdunlock(f);
		fclose(f);

		// Loading current configuration
		ncpus = _starpu_topology_get_nhwcpu(config);
		nnumas = _starpu_topology_get_nnumanodes(config);
#ifdef STARPU_USE_CUDA
		ncuda = _starpu_get_cuda_device_count();
#endif
#ifdef STARPU_USE_OPENCL
		nopencl = _starpu_opencl_get_device_count();
#endif
#ifdef STARPU_USE_MIC
		nmic = _starpu_mic_src_get_device_count();
#endif /* STARPU_USE_MIC */
#ifdef STARPU_USE_MPI_MASTER_SLAVE
		nmpi_ms = _starpu_mpi_src_get_device_count();
#endif /* STARPU_USE_MPI_MASTER_SLAVE */

		// Checking if both configurations match
		compare_value_and_recalibrate("CPUS", read_cpus, ncpus);
		compare_value_and_recalibrate("NUMA", read_numa, nnumas);
		compare_value_and_recalibrate("CUDA", read_cuda, ncuda);
		compare_value_and_recalibrate("OpenCL", read_opencl, nopencl);
		compare_value_and_recalibrate("MIC", read_mic, nmic);
		compare_value_and_recalibrate("MPI Master-Slave", read_mpi_ms, nmpi_ms);
	}
}

static void write_bus_config_file_content(void)
{
	FILE *f;
	char path[PATH_LENGTH];
	int locked;

	STARPU_ASSERT(was_benchmarked);
	get_config_path(path, sizeof(path));

	_STARPU_DEBUG("writing config to %s\n", path);

	f = fopen(path, "w+");
	STARPU_ASSERT_MSG(f, "Error when opening file (writing) '%s'", path);
	locked = _starpu_fwrlock(f) == 0;
	_starpu_fftruncate(f, 0);

	fprintf(f, "# Current configuration\n");
	fprintf(f, "%u # Number of CPUs\n", ncpus);
	fprintf(f, "%u # Number of NUMA nodes\n", nnumas);
	fprintf(f, "%u # Number of CUDA devices\n", ncuda);
	fprintf(f, "%u # Number of OpenCL devices\n", nopencl);
	fprintf(f, "%u # Number of MIC devices\n", nmic);
	fprintf(f, "%u # Number of MPI devices\n", nmpi_ms);

	if (locked)
		_starpu_fwrunlock(f);
	fclose(f);
}

static void generate_bus_config_file(void)
{
	if (!was_benchmarked)
		benchmark_all_gpu_devices();

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	/* Slaves don't write files */
	if (!_starpu_mpi_common_is_src_node())
		return;
#endif

	write_bus_config_file_content();
}
#endif /* !SIMGRID */

void _starpu_simgrid_get_platform_path(int version, char *path, size_t maxlen)
{
	if (version == 3)
		get_bus_path("platform.xml", path, maxlen);
	else
		get_bus_path("platform.v4.xml", path, maxlen);
}

#ifndef STARPU_SIMGRID
/*
 * Compute the precise PCI tree bandwidth and link shares
 *
 * We only have measurements from one leaf to another. We assume that the
 * available bandwidth is greater at lower levels, and thus measurements from
 * increasingly far GPUs provide the PCI bridges bandwidths at each level.
 *
 * The bandwidth of a PCI bridge is thus computed as the maximum of the speed
 * of the various transfers that we have achieved through it.  We thus browse
 * the PCI tree three times:
 *
 * - first through all CUDA-CUDA possible transfers to compute the maximum
 *   measured bandwidth on each PCI link and hub used for that.
 * - then through the whole tree to emit links for each PCI link and hub.
 * - then through all CUDA-CUDA possible transfers again to emit routes.
 */

#if defined(STARPU_USE_CUDA) && defined(HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX && defined(STARPU_HAVE_CUDA_MEMCPY_PEER)

/* Records, for each PCI link and hub, the maximum bandwidth seen through it */
struct pci_userdata
{
	/* Uplink max measurement */
	double bw_up;
	double bw_down;

	/* Hub max measurement */
	double bw;
};

/* Allocate a pci_userdata structure for the given object */
static void allocate_userdata(hwloc_obj_t obj)
{
	struct pci_userdata *data;

	if (obj->userdata)
		return;

	_STARPU_MALLOC(obj->userdata, sizeof(*data));
	data = obj->userdata;
	data->bw_up = 0.0;
	data->bw_down = 0.0;
	data->bw = 0.0;
}

/* Update the maximum bandwidth seen going to upstream */
static void update_bandwidth_up(hwloc_obj_t obj, double bandwidth)
{
	struct pci_userdata *data;
	if (obj->type != HWLOC_OBJ_BRIDGE && obj->type != HWLOC_OBJ_PCI_DEVICE)
		return;
	allocate_userdata(obj);

	data = obj->userdata;
	if (data->bw_up < bandwidth)
		data->bw_up = bandwidth;
}

/* Update the maximum bandwidth seen going from upstream */
static void update_bandwidth_down(hwloc_obj_t obj, double bandwidth)
{
	struct pci_userdata *data;
	if (obj->type != HWLOC_OBJ_BRIDGE && obj->type != HWLOC_OBJ_PCI_DEVICE)
		return;
	allocate_userdata(obj);

	data = obj->userdata;
	if (data->bw_down < bandwidth)
		data->bw_down = bandwidth;
}

/* Update the maximum bandwidth seen going through this Hub */
static void update_bandwidth_through(hwloc_obj_t obj, double bandwidth)
{
	struct pci_userdata *data;
	allocate_userdata(obj);

	data = obj->userdata;
	if (data->bw < bandwidth)
		data->bw = bandwidth;
}

/* find_* functions perform the first step: computing maximum bandwidths */

/* Our trafic had to go through the host, go back from target up to the host,
 * updating uplink downstream bandwidth along the way */
static void find_platform_backward_path(hwloc_obj_t obj, double bandwidth)
{
	if (!obj)
		/* Oops, we should have seen a host bridge. Well, too bad. */
		return;

	/* Update uplink bandwidth of PCI Hub */
	update_bandwidth_down(obj, bandwidth);
	/* Update internal bandwidth of PCI Hub */
	update_bandwidth_through(obj, bandwidth);

	if (obj->type == HWLOC_OBJ_BRIDGE && obj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST)
		/* Finished */
		return;

	/* Continue up */
	find_platform_backward_path(obj->parent, bandwidth);
}
/* Same, but update uplink upstream bandwidth */
static void find_platform_forward_path(hwloc_obj_t obj, double bandwidth)
{
	if (!obj)
		/* Oops, we should have seen a host bridge. Well, too bad. */
		return;

	/* Update uplink bandwidth of PCI Hub */
	update_bandwidth_up(obj, bandwidth);
	/* Update internal bandwidth of PCI Hub */
	update_bandwidth_through(obj, bandwidth);

	if (obj->type == HWLOC_OBJ_BRIDGE && obj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST)
		/* Finished */
		return;

	/* Continue up */
	find_platform_forward_path(obj->parent, bandwidth);
}

/* Find the path from obj1 through parent down to obj2 (without ever going up),
 * and update the maximum bandwidth along the path */
static int find_platform_path_down(hwloc_obj_t parent, hwloc_obj_t obj1, hwloc_obj_t obj2, double bandwidth)
{
	unsigned i;

	/* Base case, path is empty */
	if (parent == obj2)
		return 1;

	/* Try to go down from parent */
	for (i = 0; i < parent->arity; i++)
		if (parent->children[i] != obj1 && find_platform_path_down(parent->children[i], NULL, obj2, bandwidth))
		{
			/* Found it down there, update bandwidth of parent */
			update_bandwidth_down(parent->children[i], bandwidth);
			update_bandwidth_through(parent, bandwidth);
			return 1;
		}
#if HWLOC_API_VERSION >= 0x00020000
	hwloc_obj_t io;
	for (io = parent->io_first_child; io; io = io->next_sibling)
		if (io != obj1 && find_platform_path_down(io, NULL, obj2, bandwidth))
		{
			/* Found it down there, update bandwidth of parent */
			update_bandwidth_down(io, bandwidth);
			update_bandwidth_through(parent, bandwidth);
			return 1;
		}
#endif
	return 0;
}

/* Find the path from obj1 to obj2, and update the maximum bandwidth along the
 * path */
static int find_platform_path_up(hwloc_obj_t obj1, hwloc_obj_t obj2, double bandwidth)
{
	int ret;
	hwloc_obj_t parent = obj1->parent;

	if (!parent)
	{
		/* Oops, we should have seen a host bridge. Act as if we had seen it.  */
		find_platform_backward_path(obj2, bandwidth);
		return 1;
	}

	if (find_platform_path_down(parent, obj1, obj2, bandwidth))
		/* obj2 was a mere (sub)child of our parent */
		return 1;

	/* obj2 is not a (sub)child of our parent, we have to go up through the parent */
	if (parent->type == HWLOC_OBJ_BRIDGE && parent->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST)
	{
		/* We have to go up to the Host, so obj2 is not in the same PCI
		 * tree, so we're for for obj1 to Host, and just find the path
		 * from obj2 to Host too.
		 */
		find_platform_backward_path(obj2, bandwidth);

		update_bandwidth_up(parent, bandwidth);
		update_bandwidth_through(parent, bandwidth);

		return 1;
	}

	/* Not at host yet, just go up */
	ret = find_platform_path_up(parent, obj2, bandwidth);
	update_bandwidth_up(parent, bandwidth);
	update_bandwidth_through(parent, bandwidth);
	return ret;
}

/* find the path between cuda i and cuda j, and update the maximum bandwidth along the path */
static int find_platform_cuda_path(hwloc_topology_t topology, unsigned i, unsigned j, double bandwidth)
{
	hwloc_obj_t cudai, cudaj;
	cudai = hwloc_cuda_get_device_osdev_by_index(topology, i);
	cudaj = hwloc_cuda_get_device_osdev_by_index(topology, j);

	if (!cudai || !cudaj)
		return 0;

	return find_platform_path_up(cudai, cudaj, bandwidth);
}

/* emit_topology_bandwidths performs the second step: emitting link names */

/* Emit the link name of the object */
static void emit_pci_hub(FILE *f, hwloc_obj_t obj)
{
	STARPU_ASSERT(obj->type == HWLOC_OBJ_BRIDGE);
	fprintf(f, "PCI:%04x:[%02x-%02x]", obj->attr->bridge.downstream.pci.domain, obj->attr->bridge.downstream.pci.secondary_bus, obj->attr->bridge.downstream.pci.subordinate_bus);
}

static void emit_pci_dev(FILE *f, struct hwloc_pcidev_attr_s *pcidev)
{
	fprintf(f, "PCI:%04x:%02x:%02x.%1x", pcidev->domain, pcidev->bus, pcidev->dev, pcidev->func);
}

/* Emit the links of the object */
static void emit_topology_bandwidths(FILE *f, hwloc_obj_t obj, const char *Bps, const char *s)
{
	unsigned i;
	if (obj->userdata)
	{
		struct pci_userdata *data = obj->userdata;

		if (obj->type == HWLOC_OBJ_BRIDGE)
		{
			/* Uplink */
			fprintf(f, "   <link id=\"");
			emit_pci_hub(f, obj);
			fprintf(f, " up\" bandwidth=\"%f%s\" latency=\"0.000000%s\"/>\n", data->bw_up, Bps, s);
			fprintf(f, "   <link id=\"");
			emit_pci_hub(f, obj);
			fprintf(f, " down\" bandwidth=\"%f%s\" latency=\"0.000000%s\"/>\n", data->bw_down, Bps, s);

			/* PCI Switches are assumed to have infinite internal bandwidth */
			if (!obj->name || !strstr(obj->name, "Switch"))
			{
				/* We assume that PCI Hubs have double bandwidth in
				 * order to support full duplex but not more */
				fprintf(f, "   <link id=\"");
				emit_pci_hub(f, obj);
				fprintf(f, " through\" bandwidth=\"%f%s\" latency=\"0.000000%s\"/>\n", data->bw * 2, Bps, s);
			}
		}
		else if (obj->type == HWLOC_OBJ_PCI_DEVICE)
		{
			fprintf(f, "   <link id=\"");
			emit_pci_dev(f, &obj->attr->pcidev);
			fprintf(f, " up\" bandwidth=\"%f%s\" latency=\"0.000000%s\"/>\n", data->bw_up, Bps, s);
			fprintf(f, "   <link id=\"");
			emit_pci_dev(f, &obj->attr->pcidev);
			fprintf(f, " down\" bandwidth=\"%f%s\" latency=\"0.000000%s\"/>\n", data->bw_down, Bps, s);
		}
	}

	for (i = 0; i < obj->arity; i++)
		emit_topology_bandwidths(f, obj->children[i], Bps, s);
#if HWLOC_API_VERSION >= 0x00020000
	hwloc_obj_t io;
	for (io = obj->io_first_child; io; io = io->next_sibling)
		emit_topology_bandwidths(f, io, Bps, s);
#endif
}

/* emit_pci_link_* functions perform the third step: emitting the routes */

static void emit_pci_link(FILE *f, hwloc_obj_t obj, const char *suffix)
{
	if (obj->type == HWLOC_OBJ_BRIDGE)
	{
		fprintf(f, "    <link_ctn id=\"");
		emit_pci_hub(f, obj);
		fprintf(f, " %s\"/>\n", suffix);
	}
	else if (obj->type == HWLOC_OBJ_PCI_DEVICE)
	{
		fprintf(f, "    <link_ctn id=\"");
		emit_pci_dev(f, &obj->attr->pcidev);
		fprintf(f, " %s\"/>\n", suffix);
	}
}

/* Go to upstream */
static void emit_pci_link_up(FILE *f, hwloc_obj_t obj)
{
	emit_pci_link(f, obj, "up");
}

/* Go from upstream */
static void emit_pci_link_down(FILE *f, hwloc_obj_t obj)
{
	emit_pci_link(f, obj, "down");
}

/* Go through PCI hub */
static void emit_pci_link_through(FILE *f, hwloc_obj_t obj)
{
	/* We don't care about trafic going through PCI switches */
	if (obj->type == HWLOC_OBJ_BRIDGE)
	{
		if (!obj->name || !strstr(obj->name, "Switch"))
			emit_pci_link(f, obj, "through");
		else
		{
			fprintf(f, "    <!--   Switch ");
			emit_pci_hub(f, obj);
			fprintf(f, " through -->\n");
		}
	}
}

/* Our trafic has to go through the host, go back from target up to the host,
 * using uplink downstream along the way */
static void emit_platform_backward_path(FILE *f, hwloc_obj_t obj)
{
	if (!obj)
		/* Oops, we should have seen a host bridge. Well, too bad. */
		return;

	/* Go through PCI Hub */
	emit_pci_link_through(f, obj);
	/* Go through uplink */
	emit_pci_link_down(f, obj);

	if (obj->type == HWLOC_OBJ_BRIDGE && obj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST)
	{
		/* Finished, go through host */
		fprintf(f, "    <link_ctn id=\"Host\"/>\n");
		return;
	}

	/* Continue up */
	emit_platform_backward_path(f, obj->parent);
}
/* Same, but use upstream link */
static void emit_platform_forward_path(FILE *f, hwloc_obj_t obj)
{
	if (!obj)
		/* Oops, we should have seen a host bridge. Well, too bad. */
		return;

	/* Go through PCI Hub */
	emit_pci_link_through(f, obj);
	/* Go through uplink */
	emit_pci_link_up(f, obj);

	if (obj->type == HWLOC_OBJ_BRIDGE && obj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST)
	{
		/* Finished, go through host */
		fprintf(f, "    <link_ctn id=\"Host\"/>\n");
		return;
	}

	/* Continue up */
	emit_platform_forward_path(f, obj->parent);
}

/* Find the path from obj1 through parent down to obj2 (without ever going up),
 * and use the links along the path */
static int emit_platform_path_down(FILE *f, hwloc_obj_t parent, hwloc_obj_t obj1, hwloc_obj_t obj2)
{
	unsigned i;

	/* Base case, path is empty */
	if (parent == obj2)
		return 1;

	/* Try to go down from parent */
	for (i = 0; i < parent->arity; i++)
		if (parent->children[i] != obj1 && emit_platform_path_down(f, parent->children[i], NULL, obj2))
		{
			/* Found it down there, path goes through this hub */
			emit_pci_link_down(f, parent->children[i]);
			emit_pci_link_through(f, parent);
			return 1;
		}
#if HWLOC_API_VERSION >= 0x00020000
	hwloc_obj_t io;
	for (io = parent->io_first_child; io; io = io->next_sibling)
		if (io != obj1 && emit_platform_path_down(f, io, NULL, obj2))
		{
			/* Found it down there, path goes through this hub */
			emit_pci_link_down(f, io);
			emit_pci_link_through(f, parent);
			return 1;
		}
#endif
	return 0;
}

/* Find the path from obj1 to obj2, and use the links along the path */
static int emit_platform_path_up(FILE *f, hwloc_obj_t obj1, hwloc_obj_t obj2)
{
	int ret;
	hwloc_obj_t parent = obj1->parent;

	if (!parent)
	{
		/* Oops, we should have seen a host bridge. Act as if we had seen it.  */
		emit_platform_backward_path(f, obj2);
		return 1;
	}

	if (emit_platform_path_down(f, parent, obj1, obj2))
		/* obj2 was a mere (sub)child of our parent */
		return 1;

	/* obj2 is not a (sub)child of our parent, we have to go up through the parent */
	if (parent->type == HWLOC_OBJ_BRIDGE && parent->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST)
	{
		/* We have to go up to the Host, so obj2 is not in the same PCI
		 * tree, so we're for for obj1 to Host, and just find the path
		 * from obj2 to Host too.
		 */
		emit_platform_backward_path(f, obj2);
		fprintf(f, "    <link_ctn id=\"Host\"/>\n");

		emit_pci_link_up(f, parent);
		emit_pci_link_through(f, parent);

		return 1;
	}

	/* Not at host yet, just go up */
	ret = emit_platform_path_up(f, parent, obj2);
	emit_pci_link_up(f, parent);
	emit_pci_link_through(f, parent);
	return ret;
}

/* Clean our mess in the topology before destroying it */
static void clean_topology(hwloc_obj_t obj)
{
	unsigned i;
	if (obj->userdata)
	{
		free(obj->userdata);
		obj->userdata = NULL;
	}
	for (i = 0; i < obj->arity; i++)
		clean_topology(obj->children[i]);
#if HWLOC_API_VERSION >= 0x00020000
	hwloc_obj_t io;
	for (io = obj->io_first_child; io; io = io->next_sibling)
		clean_topology(io);
#endif
}
#endif

static void write_bus_platform_file_content(int version)
{
	FILE *f;
	char path[PATH_LENGTH];
	unsigned i;
	const char *speed, *flops, *Bps, *s;
	char dash;
	int locked;

	if (version == 3)
	{
		speed = "power";
		flops = "";
		Bps = "";
		s = "";
		dash = '_';
	}
	else
	{
		speed = "speed";
		flops = "f";
		Bps = "Bps";
		s = "s";
		dash = '-';
	}

	STARPU_ASSERT(was_benchmarked);

	_starpu_simgrid_get_platform_path(version, path, sizeof(path));

	_STARPU_DEBUG("writing platform to %s\n", path);

	f = fopen(path, "w+");
	if (!f)
	{
		perror("fopen write_bus_platform_file_content");
		_STARPU_DISP("path '%s'\n", path);
		fflush(stderr);
		STARPU_ABORT();
	}
	locked = _starpu_fwrlock(f) == 0;
	_starpu_fftruncate(f, 0);

	fprintf(f,
			"<?xml version='1.0'?>\n"
			"<!DOCTYPE platform SYSTEM '%s'>\n"
			" <platform version=\"%d\">\n"
			" <config id=\"General\">\n"
			"   <prop id=\"network/TCP%cgamma\" value=\"-1\"></prop>\n"
			"   <prop id=\"network/latency%cfactor\" value=\"1\"></prop>\n"
			"   <prop id=\"network/bandwidth%cfactor\" value=\"1\"></prop>\n"
			"   <prop id=\"network/weight%cS\" value=\"0.0\"></prop>\n"
			" </config>\n"
			" <AS  id=\"AS0\"  routing=\"Full\">\n"
			"   <host id=\"MAIN\" %s=\"1%s\"/>\n",
			version == 3
			? "http://simgrid.gforge.inria.fr/simgrid.dtd"
			: "http://simgrid.gforge.inria.fr/simgrid/simgrid.dtd",
			version, dash, dash, dash, dash, speed, flops);

	for (i = 0; i < ncpus; i++)
		/* TODO: host memory for out-of-core simulation */
		fprintf(f, "   <host id=\"CPU%u\" %s=\"2000000000%s\"/>\n", i, speed, flops);

	for (i = 0; i < ncuda; i++)
	{
		fprintf(f, "   <host id=\"CUDA%u\" %s=\"2000000000%s\">\n", i, speed, flops);
		fprintf(f, "     <prop id=\"memsize\" value=\"%llu\"/>\n", (unsigned long long) cuda_size[i]);
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
		fprintf(f, "     <prop id=\"memcpy_peer\" value=\"1\"/>\n");
#endif
		/* TODO: record cudadev_direct instead of assuming it's NUMA nodes */
		fprintf(f, "   </host>\n");
	}

	for (i = 0; i < nopencl; i++)
	{
		fprintf(f, "   <host id=\"OpenCL%u\" %s=\"2000000000%s\">\n", i, speed, flops);
		fprintf(f, "     <prop id=\"memsize\" value=\"%llu\"/>\n", (unsigned long long) opencl_size[i]);
		fprintf(f, "   </host>\n");
	}

	fprintf(f, "\n   <host id=\"RAM\" %s=\"1%s\"/>\n", speed, flops);

	/*
	 * Compute maximum bandwidth, taken as host bandwidth
	 */
	double max_bandwidth = 0;
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
	unsigned numa;
#endif

#ifdef STARPU_USE_CUDA
	for (i = 0; i < ncuda; i++)
	{
		for (numa = 0; numa < nnumas; numa++)
		{
			double down_bw = 1.0 / cudadev_timing_per_numa[i*STARPU_MAXNUMANODES+numa].timing_dtoh;
			double up_bw = 1.0 / cudadev_timing_per_numa[i*STARPU_MAXNUMANODES+numa].timing_htod;
			if (max_bandwidth < down_bw)
				max_bandwidth = down_bw;
			if (max_bandwidth < up_bw)
				max_bandwidth = up_bw;
		}
	}
#endif
#ifdef STARPU_USE_OPENCL
	for (i = 0; i < nopencl; i++)
	{
		for (numa = 0; numa < nnumas; numa++)
		{
			double down_bw = 1.0 / opencldev_timing_per_numa[i*STARPU_MAXNUMANODES+numa].timing_dtoh;
			double up_bw = 1.0 / opencldev_timing_per_numa[i*STARPU_MAXNUMANODES+numa].timing_htod;
			if (max_bandwidth < down_bw)
				max_bandwidth = down_bw;
			if (max_bandwidth < up_bw)
				max_bandwidth = up_bw;
		}
	}
#endif
	fprintf(f, "\n   <link id=\"Host\" bandwidth=\"%f%s\" latency=\"0.000000%s\"/>\n\n", max_bandwidth*1000000, Bps, s);

	/*
	 * OpenCL links
	 */

#ifdef STARPU_USE_OPENCL
	for (i = 0; i < nopencl; i++)
	{
		char i_name[16];
		snprintf(i_name, sizeof(i_name), "OpenCL%u", i);
		fprintf(f, "   <link id=\"RAM-%s\" bandwidth=\"%f%s\" latency=\"%f%s\"/>\n",
				i_name,
				1000000 / search_bus_best_timing(i, "OpenCL", 1), Bps,
				search_bus_best_latency(i, "OpenCL", 1)/1000000., s);
		fprintf(f, "   <link id=\"%s-RAM\" bandwidth=\"%f%s\" latency=\"%f%s\"/>\n",
				i_name,
				1000000 / search_bus_best_timing(i, "OpenCL", 0), Bps,
				search_bus_best_latency(i, "OpenCL", 0)/1000000., s);
	}
	fprintf(f, "\n");
#endif

	/*
	 * CUDA links and routes
	 */

#ifdef STARPU_USE_CUDA
	/* Write RAM/CUDA bandwidths and latencies */
	for (i = 0; i < ncuda; i++)
	{
		char i_name[16];
		snprintf(i_name, sizeof(i_name), "CUDA%u", i);
		fprintf(f, "   <link id=\"RAM-%s\" bandwidth=\"%f%s\" latency=\"%f%s\"/>\n",
				i_name,
				1000000. / search_bus_best_timing(i, "CUDA", 1), Bps,
				search_bus_best_latency(i, "CUDA", 1)/1000000., s);
		fprintf(f, "   <link id=\"%s-RAM\" bandwidth=\"%f%s\" latency=\"%f%s\"/>\n",
				i_name,
				1000000. / search_bus_best_timing(i, "CUDA", 0), Bps,
				search_bus_best_latency(i, "CUDA", 0)/1000000., s);
	}
	fprintf(f, "\n");
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	/* Write CUDA/CUDA bandwidths and latencies */
	for (i = 0; i < ncuda; i++)
	{
		unsigned j;
		char i_name[16];
		snprintf(i_name, sizeof(i_name), "CUDA%u", i);
		for (j = 0; j < ncuda; j++)
		{
			char j_name[16];
			if (j == i)
				continue;
			snprintf(j_name, sizeof(j_name), "CUDA%u", j);
			fprintf(f, "   <link id=\"%s-%s\" bandwidth=\"%f%s\" latency=\"%f%s\"/>\n",
					i_name, j_name,
					1000000. / cudadev_timing_dtod[i][j], Bps,
					cudadev_latency_dtod[i][j]/1000000., s);
		}
	}
#endif

#if defined(HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX && defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
	/* If we have enough hwloc information, write PCI bandwidths and routes */
	if (!starpu_get_env_number_default("STARPU_PCI_FLAT", 0))
	{
		hwloc_topology_t topology;
		hwloc_topology_init(&topology);
		_starpu_topology_filter(topology);
		hwloc_topology_load(topology);

		/* First find paths and record measured bandwidth along the path */
		for (i = 0; i < ncuda; i++)
		{
			unsigned j;
			for (j = 0; j < ncuda; j++)
				if (i != j)
					if (!find_platform_cuda_path(topology, i, j, 1000000. / cudadev_timing_dtod[i][j]))
					{
						clean_topology(hwloc_get_root_obj(topology));
						hwloc_topology_destroy(topology);
						goto flat_cuda;
					}
			/* Record RAM/CUDA bandwidths */
			find_platform_forward_path(hwloc_cuda_get_device_osdev_by_index(topology, i), 1000000. / search_bus_best_timing(i, "CUDA", 0));
			find_platform_backward_path(hwloc_cuda_get_device_osdev_by_index(topology, i), 1000000. / search_bus_best_timing(i, "CUDA", 1));
		}

		/* Ok, found path in all cases, can emit advanced platform routes */
		fprintf(f, "\n");
		emit_topology_bandwidths(f, hwloc_get_root_obj(topology), Bps, s);
		fprintf(f, "\n");
		for (i = 0; i < ncuda; i++)
		{
			unsigned j;
			for (j = 0; j < ncuda; j++)
				if (i != j)
				{
					fprintf(f, "   <route src=\"CUDA%u\" dst=\"CUDA%u\" symmetrical=\"NO\">\n", i, j);
					fprintf(f, "    <link_ctn id=\"CUDA%u-CUDA%u\"/>\n", i, j);
					emit_platform_path_up(f,
							hwloc_cuda_get_device_osdev_by_index(topology, i),
							hwloc_cuda_get_device_osdev_by_index(topology, j));
					fprintf(f, "   </route>\n");
				}

			fprintf(f, "   <route src=\"CUDA%u\" dst=\"RAM\" symmetrical=\"NO\">\n", i);
			fprintf(f, "    <link_ctn id=\"CUDA%u-RAM\"/>\n", i);
			emit_platform_forward_path(f, hwloc_cuda_get_device_osdev_by_index(topology, i));
			fprintf(f, "   </route>\n");

			fprintf(f, "   <route src=\"RAM\" dst=\"CUDA%u\" symmetrical=\"NO\">\n", i);
			fprintf(f, "    <link_ctn id=\"RAM-CUDA%u\"/>\n", i);
			emit_platform_backward_path(f, hwloc_cuda_get_device_osdev_by_index(topology, i));
			fprintf(f, "   </route>\n");
		}

		clean_topology(hwloc_get_root_obj(topology));
		hwloc_topology_destroy(topology);
	}
	else
	{
flat_cuda:
#else
	{
#endif
		/* If we don't have enough hwloc information, write trivial routes always through host */
		for (i = 0; i < ncuda; i++)
		{
			char i_name[16];
			snprintf(i_name, sizeof(i_name), "CUDA%u", i);
			fprintf(f, "   <route src=\"RAM\" dst=\"%s\" symmetrical=\"NO\"><link_ctn id=\"RAM-%s\"/><link_ctn id=\"Host\"/></route>\n", i_name, i_name);
			fprintf(f, "   <route src=\"%s\" dst=\"RAM\" symmetrical=\"NO\"><link_ctn id=\"%s-RAM\"/><link_ctn id=\"Host\"/></route>\n", i_name, i_name);
		}
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
		for (i = 0; i < ncuda; i++)
		{
                        unsigned j;
                        char i_name[16];
                        snprintf(i_name, sizeof(i_name), "CUDA%u", i);
			for (j = 0; j < ncuda; j++)
			{
				char j_name[16];
				if (j == i)
					continue;
                                snprintf(j_name, sizeof(j_name), "CUDA%u", j);
				fprintf(f, "   <route src=\"%s\" dst=\"%s\" symmetrical=\"NO\"><link_ctn id=\"%s-%s\"/><link_ctn id=\"Host\"/></route>\n", i_name, j_name, i_name, j_name);
			}
		}
#endif
	} /* defined(STARPU_HAVE_HWLOC) && defined(STARPU_HAVE_CUDA_MEMCPY_PEER) */
	fprintf(f, "\n");
#endif /* STARPU_USE_CUDA */

	/*
	 * OpenCL routes
	 */

#ifdef STARPU_USE_OPENCL
	for (i = 0; i < nopencl; i++)
	{
		char i_name[16];
		snprintf(i_name, sizeof(i_name), "OpenCL%u", i);
		fprintf(f, "   <route src=\"RAM\" dst=\"%s\" symmetrical=\"NO\"><link_ctn id=\"RAM-%s\"/><link_ctn id=\"Host\"/></route>\n", i_name, i_name);
		fprintf(f, "   <route src=\"%s\" dst=\"RAM\" symmetrical=\"NO\"><link_ctn id=\"%s-RAM\"/><link_ctn id=\"Host\"/></route>\n", i_name, i_name);
	}
#endif

	fprintf(f,
		" </AS>\n"
		" </platform>\n"
	       );

	if (locked)
		_starpu_fwrunlock(f);
	fclose(f);

}

static void generate_bus_platform_file(void)
{
	if (!was_benchmarked)
		benchmark_all_gpu_devices();

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	/* Slaves don't write files */
	if (!_starpu_mpi_common_is_src_node())
		return;
#endif

	write_bus_platform_file_content(3);
	write_bus_platform_file_content(4);
}

static void check_bus_platform_file(void)
{
	int res;

	char path[PATH_LENGTH];
	_starpu_simgrid_get_platform_path(4, path, sizeof(path));

	res = access(path, F_OK);

	if (!res)
	{
		_starpu_simgrid_get_platform_path(3, path, sizeof(path));
		res = access(path, F_OK);
	}

	if (res)
	{
		/* File does not exist yet */
		generate_bus_platform_file();
	}
}

/*
 *	Generic
 */

static void _starpu_bus_force_sampling(void)
{
	_STARPU_DEBUG("Force bus sampling ...\n");
	_starpu_create_sampling_directory_if_needed();

	generate_bus_affinity_file();
	generate_bus_latency_file();
	generate_bus_bandwidth_file();
	generate_bus_config_file();
	generate_bus_platform_file();
}
#endif /* !SIMGRID */

void _starpu_load_bus_performance_files(void)
{
	_starpu_create_sampling_directory_if_needed();

	struct _starpu_machine_config * config = _starpu_get_machine_config();
	nnumas = _starpu_topology_get_nnumanodes(config);
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_SIMGRID)
	ncuda = _starpu_get_cuda_device_count();
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_USE_SIMGRID)
	nopencl = _starpu_opencl_get_device_count();
#endif
#if defined(STARPU_USE_MPI_MASTER_SLAVE) || defined(STARPU_USE_SIMGRID)
	nmpi_ms = _starpu_mpi_src_get_device_count();
#endif
#if defined(STARPU_USE_MIC) || defined(STARPU_USE_SIMGRID)
	nmic = _starpu_mic_src_get_device_count();
#endif

#ifndef STARPU_SIMGRID
	check_bus_config_file();
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	/* be sure that master wrote the perf files */
	_starpu_mpi_common_barrier();
#endif

#ifndef STARPU_SIMGRID
	load_bus_affinity_file();
#endif
	load_bus_latency_file();
	load_bus_bandwidth_file();
#ifndef STARPU_SIMGRID
	check_bus_platform_file();
#endif
}

/* (in MB/s) */
double starpu_transfer_bandwidth(unsigned src_node, unsigned dst_node)
{
	return bandwidth_matrix[src_node][dst_node];
}

/* (in µs) */
double starpu_transfer_latency(unsigned src_node, unsigned dst_node)
{
	return latency_matrix[src_node][dst_node];
}

/* (in µs) */
double starpu_transfer_predict(unsigned src_node, unsigned dst_node, size_t size)
{
	if (src_node == dst_node)
		return 0;

	double bandwidth = bandwidth_matrix[src_node][dst_node];
	double latency = latency_matrix[src_node][dst_node];
	struct _starpu_machine_topology *topology = &_starpu_get_machine_config()->topology;
#if 0
	int busid = starpu_bus_get_id(src_node, dst_node);
	int direct = starpu_bus_get_direct(busid);
#endif
	float ngpus = topology->ncudagpus+topology->nopenclgpus;
#ifdef STARPU_DEVEL
#warning FIXME: ngpus should not be used e.g. for slow disk transfers...
#endif

#if 0
	/* Ideally we should take into account that some GPUs are directly
	 * connected through a PCI switch, which has less contention that the
	 * Host bridge, but doing that seems to *decrease* performance... */
	if (direct)
	{
		float neighbours = starpu_bus_get_ngpus(busid);
		/* Count transfers of these GPUs, and count transfers between
		 * other GPUs and these GPUs */
		ngpus = neighbours + (ngpus - neighbours) * neighbours / ngpus;
	}
#endif


	return latency + (size/bandwidth)*2*ngpus;
}

/* calculate save bandwidth and latency */
/* bandwidth in MB/s - latency in µs */
void _starpu_save_bandwidth_and_latency_disk(double bandwidth_write, double bandwidth_read, double latency_write, double latency_read, unsigned node, const char *name)
{
	unsigned int i, j;
	double slowness_disk_between_main_ram, slowness_main_ram_between_node;
	int print_stats = starpu_get_env_number_default("STARPU_BUS_STATS", 0);

	if (print_stats)
	{
		fprintf(stderr, "\n#---------------------\n");
		fprintf(stderr, "Data transfer speed for %s (node %u):\n", name, node);
	}

	/* save bandwith */
	for(i = 0; i < STARPU_MAXNODES; ++i)
	{
		for(j = 0; j < STARPU_MAXNODES; ++j)
		{
			if (i == j && j == node) /* source == destination == node */
			{
				bandwidth_matrix[i][j] = 0;
			}
			else if (i == node) /* source == disk */
			{
				/* convert in slowness */
				if(bandwidth_read != 0)
					slowness_disk_between_main_ram = 1/bandwidth_read;
				else
					slowness_disk_between_main_ram = 0;

				if(bandwidth_matrix[STARPU_MAIN_RAM][j] != 0)
					slowness_main_ram_between_node = 1/bandwidth_matrix[STARPU_MAIN_RAM][j];
				else
					slowness_main_ram_between_node = 0;

				bandwidth_matrix[i][j] = 1/(slowness_disk_between_main_ram+slowness_main_ram_between_node);

				if (!isnan(bandwidth_matrix[i][j]) && print_stats)
					fprintf(stderr,"%u -> %u: %.0f MB/s\n", i, j, bandwidth_matrix[i][j]);
			}
			else if (j == node) /* destination == disk */
			{
				/* convert in slowness */
				if(bandwidth_write != 0)
					slowness_disk_between_main_ram = 1/bandwidth_write;
				else
					slowness_disk_between_main_ram = 0;

				if(bandwidth_matrix[i][STARPU_MAIN_RAM] != 0)
					slowness_main_ram_between_node = 1/bandwidth_matrix[i][STARPU_MAIN_RAM];
				else
					slowness_main_ram_between_node = 0;

				bandwidth_matrix[i][j] = 1/(slowness_disk_between_main_ram+slowness_main_ram_between_node);

				if (!isnan(bandwidth_matrix[i][j]) && print_stats)
					fprintf(stderr,"%u -> %u: %.0f MB/s\n", i, j, bandwidth_matrix[i][j]);
			}
			else if (j > node || i > node) /* not affected by the node */
			{
				bandwidth_matrix[i][j] = NAN;
			}
		}
	}

	/* save latency */
	for(i = 0; i < STARPU_MAXNODES; ++i)
	{
		for(j = 0; j < STARPU_MAXNODES; ++j)
		{
			if (i == j && j == node) /* source == destination == node */
			{
				latency_matrix[i][j] = 0;
			}
			else if (i == node) /* source == disk */
			{
				latency_matrix[i][j] = (latency_write+latency_matrix[STARPU_MAIN_RAM][j]);

				if (!isnan(latency_matrix[i][j]) && print_stats)
					fprintf(stderr,"%u -> %u: %.0f us\n", i, j, latency_matrix[i][j]);
			}
			else if (j == node) /* destination == disk */
			{
				latency_matrix[i][j] = (latency_read+latency_matrix[i][STARPU_MAIN_RAM]);

				if (!isnan(latency_matrix[i][j]) && print_stats)
					fprintf(stderr,"%u -> %u: %.0f us\n", i, j, latency_matrix[i][j]);
			}
			else if (j > node || i > node) /* not affected by the node */
			{
				latency_matrix[i][j] = NAN;
			}
		}
	}

	if (print_stats)
		fprintf(stderr, "\n#---------------------\n");
}
