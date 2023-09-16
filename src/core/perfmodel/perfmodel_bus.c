/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <drivers/tcpip/driver_tcpip_common.h>
#include <datawizard/memory_nodes.h>

#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifdef STARPU_HAVE_LIBNVIDIA_ML
#include <hwloc/nvml.h>
#endif
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif

#if HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX
#include <hwloc/cuda.h>
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
#include <mpi.h>
#endif

#define SIZE	(32*1024*1024*sizeof(char))
#define NITER	32

#ifndef STARPU_SIMGRID
static void _starpu_bus_force_sampling(int location);
#endif

/* timing is in µs per byte (i.e. slowness, inverse of bandwidth) */
struct dev_timing
{
	int numa_id;
	int numa_distance;
	double timing_htod;
	double latency_htod;
	double timing_dtoh;
	double latency_dtoh;
};

static double bandwidth_matrix[STARPU_MAXNODES][STARPU_MAXNODES]; /* MB/s */
static double latency_matrix[STARPU_MAXNODES][STARPU_MAXNODES]; /* µs */
static unsigned was_benchmarked = 0;
#ifndef STARPU_SIMGRID
static unsigned ncpus = 0;
#endif
static unsigned nmem[STARPU_NRAM];
#define nnumas (nmem[STARPU_CPU_RAM])
#define ncuda (nmem[STARPU_CUDA_RAM])
#define nhip (nmem[STARPU_HIP_RAM])
#define nopencl (nmem[STARPU_OPENCL_RAM])
#define nmpims (nmem[STARPU_MPI_MS_RAM])
#define ntcpip_ms (nmem[STARPU_TCPIP_MS_RAM])

#ifndef STARPU_SIMGRID
/* Benchmarking the performance of the bus */

static double numa_latency[STARPU_MAXNUMANODES][STARPU_MAXNUMANODES];
static double numa_timing[STARPU_MAXNUMANODES][STARPU_MAXNUMANODES];

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_HIP) || defined(STARPU_USE_OPENCL)
static int gpu_numa[STARPU_NRAM][STARPU_NMAXDEVS]; /* hwloc NUMA logical ID */
#endif
#endif

/* preference order of NUMA nodes (logical indexes) */
static unsigned affinity_matrix[STARPU_NRAM][STARPU_NMAXDEVS][STARPU_MAXNUMANODES];

#ifndef STARPU_SIMGRID
#define DEV_MAXLENGTH 256
static char device_name[STARPU_NRAM][STARPU_NMAXDEVS][DEV_MAXLENGTH];
static uint64_t device_memory[STARPU_NRAM][STARPU_NMAXDEVS];
static int device_peer_access[STARPU_NRAM][STARPU_MAXNODES][STARPU_MAXNODES];
static double timing_dtod[STARPU_NRAM][STARPU_NMAXDEVS][STARPU_NMAXDEVS];
static double latency_dtod[STARPU_NRAM][STARPU_NMAXDEVS][STARPU_NMAXDEVS];

static struct dev_timing timing_per_numa[STARPU_NRAM][STARPU_NMAXDEVS][STARPU_MAXNUMANODES];
#endif

#ifdef STARPU_HAVE_HWLOC
static hwloc_topology_t hwtopology;
#if HAVE_DECL_HWLOC_DISTANCES_OBJ_PAIR_VALUES
static struct hwloc_distances_s *numa_distances;
#endif

hwloc_topology_t _starpu_perfmodel_get_hwtopology()
{
	return hwtopology;
}

static int find_cpu_from_numa_node(unsigned numa_id)
{
	hwloc_obj_t obj = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa_id);

	if (obj)
	{
#if HWLOC_API_VERSION >= 0x00020000
		/* From hwloc 2.0, NUMAnode objects do not contain CPUs, they
		 * are contained in a group which contain the CPUs. */
		obj = obj->parent;
#endif
	}
	else
	{
		/* No such NUMA node, probably hwloc 1.x with no NUMA
		 * node, just take one CPU from the whole system */
		obj = hwloc_get_root_obj(hwtopology);
	}

	STARPU_ASSERT(obj);
	hwloc_obj_t current = obj;

	while (current->type != HWLOC_OBJ_PU)
	{
		current = current->first_child;

		/* If we don't find a "PU" obj before the leave, perhaps we are
		 * just not allowed to use it. */
		if (!current)
			return -1;
	}

	STARPU_ASSERT(current->type == HWLOC_OBJ_PU);

	return current->logical_index;
}
#endif

#if (defined(STARPU_USE_CUDA) || defined(STARPU_USE_HIP) || defined(STARPU_USE_OPENCL)) && !defined(STARPU_SIMGRID)

static void set_numa_distance(int dev, unsigned numa, enum starpu_worker_archtype arch, struct dev_timing *dev_timing_per_cpu)
{
	/* A priori we don't know the distance */
	dev_timing_per_cpu->numa_distance = -1;

#ifdef STARPU_HAVE_HWLOC
	if (nnumas <= 1)
		return;

	if (!starpu_driver_info[arch].get_hwloc_obj)
		return;

	hwloc_obj_t obj = starpu_driver_info[arch].get_hwloc_obj(hwtopology, dev);
	if (!obj)
		return;

	hwloc_obj_t numa_obj = _starpu_numa_get_obj(obj);
	if (!numa_obj)
		return;

	if (numa_obj->logical_index == numa)
	{
		_STARPU_DEBUG("GPU is on NUMA %d, distance zero\n", numa);
		dev_timing_per_cpu->numa_distance = 0;
		return;
	}

#if HAVE_DECL_HWLOC_DISTANCES_OBJ_PAIR_VALUES
	if (!numa_distances)
		return;

	hwloc_obj_t drive_numa_obj = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa);
	hwloc_uint64_t gpu2drive, drive2gpu;
	if (!drive_numa_obj)
		return;

	_STARPU_DEBUG("GPU is on NUMA %d vs %d\n", numa_obj->logical_index, numa);
	if (hwloc_distances_obj_pair_values(numa_distances, numa_obj, drive_numa_obj, &gpu2drive, &drive2gpu) == 0)
	{
		_STARPU_DEBUG("got distance G2H %lu H2G %lu\n", (unsigned long) gpu2drive, (unsigned long) drive2gpu);
		dev_timing_per_cpu->numa_distance = (gpu2drive + drive2gpu) / 2;
	}
#endif
#endif
}

/* TODO: factorize MPI_MS and TCPIP_MS. Will probably need to introduce a method
 * for MPI_Barrier, and for determining which combinations should be measured. */

static void measure_bandwidth_between_host_and_dev_on_numa(int dev, enum starpu_node_kind kind, unsigned numa, int cpu,
							   struct dev_timing *dev_timing_per_cpu,
							   uint64_t *dev_size,
							   char dev_name[DEV_MAXLENGTH])
{
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);
	size_t size = SIZE;
	const struct _starpu_node_ops *ops = starpu_memory_driver_info[kind].ops;

	/* Initialize context on the device */
	ops->init_device(dev);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* Get the maximum size which can be allocated on the device */
        *dev_size = ops->total_memory(dev);
	uint64_t max_size = ops->max_memory(dev);
	if (size > max_size/4) size = max_size/4;

	ops->device_name(dev, dev_name, DEV_MAXLENGTH);

#ifdef STARPU_USE_OPENCL
	if (kind == STARPU_OPENCL_RAM && _starpu_opencl_get_device_type(dev) == CL_DEVICE_TYPE_CPU)
	{
		/* Let's not use too much RAM when running OpenCL on a CPU: it
		 * would make the OS swap like crazy. */
		size /= 2;
	}
#endif

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* Allocate a buffer on the device */
	uintptr_t d_buffer;
	d_buffer = ops->malloc_on_device(dev, size, 0);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* Allocate a buffer on the host */
	unsigned char *h_buffer;
#if defined(STARPU_HAVE_HWLOC)
	if (nnumas > 1)
	{
		/* different NUMA nodes available */
		hwloc_obj_t obj = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa);
		STARPU_ASSERT(obj);
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
#if defined(STARPU_USE_CUDA)
		if (kind == STARPU_CUDA_RAM)
			cudaHostRegister((void *)h_buffer, size, 0);
#endif
#if defined(STARPU_USE_HIP)
		if (kind == STARPU_HIP_RAM)
			hipHostRegister((void *)h_buffer, size, 0);
#endif
	}

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	/* Fill them */
	memset(h_buffer, 0, size);

	/* hack to avoid third party libs to rebind threads */
	_starpu_bind_thread_on_cpu(cpu, STARPU_NOWORKERID, NULL);

	unsigned iter;
	double timing;
	double start;
	double end;

	/* Measure upload bandwidth */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		ops->copy_data_from[STARPU_CPU_RAM]((uintptr_t)h_buffer, 0, cpu, d_buffer, 0, dev, size, NULL);
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu->timing_htod = timing/NITER/size;

	/* Measure download bandwidth */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		ops->copy_data_to[STARPU_CPU_RAM](d_buffer, 0, dev, (uintptr_t)h_buffer, 0, cpu, size, NULL);
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu->timing_dtoh = timing/NITER/size;

	/* Measure upload latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		ops->copy_data_from[STARPU_CPU_RAM]((uintptr_t)h_buffer, 0, cpu, d_buffer, 0, dev, 1, NULL);
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu->latency_htod = timing/NITER;

	/* Measure download latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		ops->copy_data_to[STARPU_CPU_RAM](d_buffer, 0, dev, (uintptr_t)h_buffer, 0, cpu, 1, NULL);
	}
	end = starpu_timing_now();
	timing = end - start;

	dev_timing_per_cpu->latency_dtoh = timing/NITER;

	/* Free buffers */
#if defined(STARPU_USE_CUDA)
	if (kind == STARPU_CUDA_RAM)
		cudaHostUnregister(h_buffer);
#endif
#if defined(STARPU_USE_HIP)
	if (kind == STARPU_HIP_RAM)
		hipHostUnregister(h_buffer);
#endif
#if defined(STARPU_HAVE_HWLOC)
	if (nnumas > 1)
	{
		/* different NUMA nodes available */
		hwloc_free(hwtopology, h_buffer, size);
	}
	else
#endif
	{
		free(h_buffer);
	}
	ops->free_on_device(dev, d_buffer, size, 0);
	ops->reset_device(dev);
}

static void measure_bandwidth_between_dev_and_dev(int src, int dst, enum starpu_node_kind kind,
						  double *timingr, double *latencyr,
						  int peer_access[STARPU_MAXNODES][STARPU_MAXNODES],
						  uint64_t dev_memory[STARPU_NMAXDEVS])
{
	size_t size = SIZE;

	const struct _starpu_node_ops *src_ops = starpu_memory_driver_info[kind].ops;
	const struct _starpu_node_ops *dst_ops = starpu_memory_driver_info[kind].ops;

	/* Initialize context on the source */
	src_ops->set_device(src);
	if (src_ops->try_enable_peer_access)
		peer_access[src][dst] = src_ops->try_enable_peer_access(src, dst);
	else
		peer_access[src][dst] = 0;

	/* Initialize context on the destination */
	dst_ops->set_device(dst);
	if (dst_ops->try_enable_peer_access)
		peer_access[dst][src] = dst_ops->try_enable_peer_access(dst, src);
	else
		peer_access[dst][src] = 0;

	/* Check for peer access and return early if it's not supported */
	if (peer_access[dst][src])
	{
		_STARPU_DISP("GPU-Direct %d -> %d\n", src, dst);
	}
	else
	{
		_STARPU_DISP("No GPU-Direct %d -> %d\n", src, dst);
		return;
	}

	/* Get the maximum size which can be allocated on the device */
	if (size > dev_memory[src]/4) size = dev_memory[src];
	if (size > dev_memory[dst]/4) size = dev_memory[dst];

	/* Allocate a buffer on the device */
	uintptr_t s_buffer;
	s_buffer = src_ops->malloc_on_device(src, size, 0);

	/* Allocate a buffer on the device */
	uintptr_t d_buffer;
	d_buffer = dst_ops->malloc_on_device(dst, size, 0);

	unsigned iter;
	double timing;
	double start;
	double end;

	/* Measure upload bandwidth */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		src_ops->copy_data_to[kind](s_buffer, 0, src, d_buffer, 0, dst, size, NULL);
	}
	end = starpu_timing_now();
	timing = end - start;

	*timingr = timing/NITER/size;

	/* Measure upload latency */
	start = starpu_timing_now();
	for (iter = 0; iter < NITER; iter++)
	{
		src_ops->copy_data_to[kind](s_buffer, 0, src, d_buffer, 0, dst, 1, NULL);
	}
	end = starpu_timing_now();
	timing = end - start;

	*latencyr = timing/NITER;

	/* Free buffers */
	src_ops->free_on_device(src, s_buffer, size, 0);
	dst_ops->free_on_device(dst, d_buffer, size, 0);
	src_ops->reset_device(src);

#ifdef STARPU_VERBOSE
	double bandwidth_dtod = 1/(*timingr);
	_STARPU_DISP("(%10s) BANDWIDTH GPU %d GPU %d - dtod %.0fMB/s\n", starpu_memory_driver_info[kind].name_upper, src, dst, bandwidth_dtod);
#endif
}

static void measure_bandwidth_between_host_and_dev(int dev, struct dev_timing dev_timing_per_numa[STARPU_NMAXDEVS][STARPU_MAXNUMANODES], enum starpu_node_kind type)
{
	enum starpu_worker_archtype arch = starpu_memory_node_get_worker_archtype(type);

	/* We measure the bandwidth between each GPU and each NUMA node */
	unsigned numa_id;
	for (numa_id = 0; numa_id < nnumas; numa_id++)
	{
		/* Store STARPU_memnode for later */
		dev_timing_per_numa[dev][numa_id].numa_id = numa_id;

		/* Chose one CPU connected to this NUMA node */
		int cpu_id = 0;
#ifdef STARPU_HAVE_HWLOC
		cpu_id = find_cpu_from_numa_node(numa_id);
#endif
		if (cpu_id < 0)
			continue;

		_STARPU_DISP("with NUMA %d...\n", numa_id);

		/* Check hwloc location of GPU */
		set_numa_distance(dev, numa_id, arch, &dev_timing_per_numa[dev][numa_id]);

		if (starpu_memory_driver_info[type].ops->calibrate_bus)
			measure_bandwidth_between_host_and_dev_on_numa(dev, type, numa_id, cpu_id,
								       &dev_timing_per_numa[dev][numa_id],
								       &device_memory[type][dev],
								       device_name[type][dev]);
	}
	/* TODO: also measure the available aggregated bandwidth on a NUMA node, and through the interconnect */

#if defined(STARPU_HAVE_HWLOC)
	hwloc_obj_t obj = NULL;

	if (starpu_driver_info[arch].get_hwloc_obj)
		obj = starpu_driver_info[arch].get_hwloc_obj(hwtopology, dev);
	if (obj)
		obj = _starpu_numa_get_obj(obj);
	if (obj)
		gpu_numa[type][dev] = obj->logical_index;
	else
#endif
		gpu_numa[type][dev] = -1;

#ifdef STARPU_VERBOSE
	for (numa_id = 0; numa_id < nnumas; numa_id++)
	{
		double bandwidth_dtoh = dev_timing_per_numa[dev][numa_id].timing_dtoh;
		double bandwidth_htod = dev_timing_per_numa[dev][numa_id].timing_htod;

		double bandwidth_sum2 = bandwidth_dtoh*bandwidth_dtoh + bandwidth_htod*bandwidth_htod;

		_STARPU_DISP("(%10s) BANDWIDTH GPU %d NUMA %u - htod %.0fMB/s - dtoh %.0fMB/s - %.0fMB/s\n", starpu_memory_driver_info[type].name_upper, dev, numa_id, 1/bandwidth_htod, 1/bandwidth_dtoh, 1/sqrt(bandwidth_sum2));
	}
#endif
}
#endif /* defined(STARPU_USE_CUDA) || defined(STARPU_USE_HIP) || defined(STARPU_USE_OPENCL) */

#if !defined(STARPU_SIMGRID)
static void measure_bandwidth_latency_between_numa(int numa_src, int numa_dst, double *timing_nton, double *latency_nton)
{
#if defined(STARPU_HAVE_HWLOC)
	if (nnumas > 1)
	{
		/* different NUMA nodes available */
		double start, end, timing;
		unsigned iter;

		/* Chose one CPU connected to this NUMA node */
		int cpu_id = 0;
		cpu_id = find_cpu_from_numa_node(numa_src);
		if (cpu_id < 0)
			/* We didn't find a CPU attached to the numa_src NUMA nodes */
			goto no_calibration;

		_starpu_bind_thread_on_cpu(cpu_id, STARPU_NOWORKERID, NULL);

		unsigned char *h_buffer;
		hwloc_obj_t obj_src = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa_src);
		STARPU_ASSERT(obj_src);
#if HWLOC_API_VERSION >= 0x00020000
		h_buffer = hwloc_alloc_membind(hwtopology, SIZE, obj_src->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
#else
		h_buffer = hwloc_alloc_membind_nodeset(hwtopology, SIZE, obj_src->nodeset, HWLOC_MEMBIND_BIND, 0);
#endif

		unsigned char *d_buffer;
		hwloc_obj_t obj_dst = hwloc_get_obj_by_type(hwtopology, HWLOC_OBJ_NUMANODE, numa_dst);
		STARPU_ASSERT(obj_dst);
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

		*timing_nton = timing/NITER/SIZE;

		start = starpu_timing_now();
		for (iter = 0; iter < NITER; iter++)
		{
			memcpy(d_buffer, h_buffer, 1);
		}
		end = starpu_timing_now();
		timing = end - start;

		*latency_nton = timing/NITER;

		hwloc_free(hwtopology, h_buffer, SIZE);
		hwloc_free(hwtopology, d_buffer, SIZE);
	}
	else
no_calibration:
#endif
	{
		/* Cannot make a real calibration */
		numa_timing[numa_src][numa_dst] = 0.01;
		numa_latency[numa_src][numa_dst] = 0;
	}
}
#endif

static void benchmark_all_memory_nodes(void)
{
#ifdef STARPU_SIMGRID
	_STARPU_DISP("Can not measure bus in simgrid mode, please run starpu_calibrate_bus in non-simgrid mode to make sure the bus performance model was calibrated\n");
	STARPU_ABORT();
#else /* !SIMGRID */
	unsigned i, j;

	_STARPU_DEBUG("Benchmarking the speed of the bus\n");

#ifdef STARPU_DEVEL
#warning FIXME: when running several StarPU processes on the same node (MPI rank per numa), we need to use a lock to avoid concurrent benchmarking.
#endif

#ifdef STARPU_HAVE_HWLOC
	int ret;
	ret  = hwloc_topology_init(&hwtopology);
	STARPU_ASSERT_MSG(ret == 0, "Could not initialize Hwloc topology (%s)\n", strerror(errno));
	_starpu_topology_filter(hwtopology);
	ret = hwloc_topology_load(hwtopology);
	STARPU_ASSERT_MSG(ret == 0, "Could not load Hwloc topology (%s)\n", strerror(errno));
#if HAVE_DECL_HWLOC_DISTANCES_OBJ_PAIR_VALUES
	unsigned n = 1;
	hwloc_distances_get_by_name(hwtopology, "NUMALatency", &n, &numa_distances, 0);
	if (!n)
		numa_distances = NULL;
#endif
#endif

#ifdef STARPU_HAVE_HWLOC
	hwloc_bitmap_t former_cpuset = hwloc_bitmap_alloc();
	hwloc_get_cpubind(hwtopology, former_cpuset, HWLOC_CPUBIND_THREAD);
#elif defined(__linux__)
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

	for (i = 0; i < nnumas; i++)
		for (j = 0; j < nnumas; j++)
			if (i != j)
			{
				_STARPU_DISP("NUMA %d -> %d...\n", i, j);
				measure_bandwidth_latency_between_numa(i, j, &numa_timing[i][j], &numa_latency[i][j]);
			}

#ifndef STARPU_SIMGRID
	struct _starpu_machine_topology *topology = &_starpu_get_machine_config()->topology;
	enum starpu_node_kind type;
	for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; ++type)
	{
		if (starpu_memory_driver_info[type].ops->calibrate_bus)
		{
			enum starpu_worker_archtype arch = starpu_memory_node_get_worker_archtype(type);
			const char *type_str = starpu_worker_get_type_as_string(arch);
			nmem[type] = topology->nhwdevices[arch];
			for (i = 0; i < nmem[type]; i++)
			{
				_STARPU_DISP("%s %u...\n", type_str, i);
				/* measure bandwidth between Host and Device i */
				measure_bandwidth_between_host_and_dev(i, timing_per_numa[type], type);
			}
			for (i = 0; i < nmem[type]; i++)
			{
				for (j = 0; j < nmem[type]; j++)
					if (i != j)
					{
						_STARPU_DISP("%s %u -> %u...\n", type_str, i, j);
						measure_bandwidth_between_dev_and_dev(i, j, type,
										      &timing_dtod[type][i][j],
										      &latency_dtod[type][i][j],
										      device_peer_access[type],
										      device_memory[type]);
					}
			}

		}
	}
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	double mpi_time_device_to_device[STARPU_MAXMPIDEVS][STARPU_MAXMPIDEVS] = {{0.0}};
	double mpi_latency_device_to_device[STARPU_MAXMPIDEVS][STARPU_MAXMPIDEVS] = {{0.0}};
	/* FIXME: rather make _starpu_mpi_common_measure_bandwidth_latency directly fill timing_per_numa */
	_starpu_mpi_common_measure_bandwidth_latency(mpi_time_device_to_device, mpi_latency_device_to_device);
	for (i = 0; i < nmpims; i++)
	{
		for (j = 0; j < nnumas; j++)
		{
			timing_per_numa[STARPU_MPI_MS_RAM][i][j].numa_id = j;
			timing_per_numa[STARPU_MPI_MS_RAM][i][j].numa_distance = -1;
			timing_per_numa[STARPU_MPI_MS_RAM][i][j].timing_htod = mpi_time_device_to_device[0][i+1];
			timing_per_numa[STARPU_MPI_MS_RAM][i][j].latency_htod = mpi_latency_device_to_device[0][i+1];
			timing_per_numa[STARPU_MPI_MS_RAM][i][j].timing_dtoh = mpi_time_device_to_device[i+1][0];
			timing_per_numa[STARPU_MPI_MS_RAM][i][j].latency_dtoh = mpi_latency_device_to_device[i+1][0];
		}
		for (j = 0; j < nmpims; j++)
		{
			timing_dtod[STARPU_MPI_MS_RAM][i][j] = mpi_time_device_to_device[i+1][j+1];
		}
	}
#endif /* STARPU_USE_MPI_MASTER_SLAVE */

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
	double tcpip_time_device_to_device[STARPU_MAXTCPIPDEVS][STARPU_MAXTCPIPDEVS] = {{0.0}};
	double tcpip_latency_device_to_device[STARPU_MAXTCPIPDEVS][STARPU_MAXTCPIPDEVS] = {{0.0}};
	/* FIXME: rather make _starpu_mpi_common_measure_bandwidth_latency directly fill timing_per_numa */
	_starpu_tcpip_common_measure_bandwidth_latency(tcpip_time_device_to_device, tcpip_latency_device_to_device);
	for (i = 0; i < ntcpip_ms; i++)
	{
		for (j = 0; j < nnumas; j++)
		{
			timing_per_numa[STARPU_TCPIP_MS_RAM][i][j].numa_id = j;
			timing_per_numa[STARPU_TCPIP_MS_RAM][i][j].numa_distance = -1;
			timing_per_numa[STARPU_TCPIP_MS_RAM][i][j].timing_htod = tcpip_time_device_to_device[0][i+1];
			timing_per_numa[STARPU_TCPIP_MS_RAM][i][j].latency_htod = tcpip_latency_device_to_device[0][i+1];
			timing_per_numa[STARPU_TCPIP_MS_RAM][i][j].timing_dtoh = tcpip_time_device_to_device[i+1][0];
			timing_per_numa[STARPU_TCPIP_MS_RAM][i][j].latency_dtoh = tcpip_latency_device_to_device[i+1][0];
		}
		for (j = 0; j < ntcpip_ms; j++)
		{
			timing_dtod[STARPU_TCPIP_MS_RAM][i][j] = tcpip_time_device_to_device[i+1][j+1];
		}
	}
#endif /* STARPU_USE_TCPIP_MASTER_SLAVE */

#ifdef STARPU_HAVE_HWLOC
	hwloc_set_cpubind(hwtopology, former_cpuset, HWLOC_CPUBIND_THREAD);
	hwloc_bitmap_free(former_cpuset);
#elif defined(__linux__)
	/* Restore the former affinity */
	ret = sched_setaffinity(0, sizeof(former_process_affinity), &former_process_affinity);
	if (ret)
	{
		perror("sched_setaffinity");
		STARPU_ABORT();
	}
#endif

#ifdef STARPU_HAVE_HWLOC
#if HAVE_DECL_HWLOC_DISTANCES_OBJ_PAIR_VALUES
	if (numa_distances)
		hwloc_distances_release(hwtopology, numa_distances);
	numa_distances = NULL;
#endif
	hwloc_topology_destroy(hwtopology);
#endif

	_STARPU_DEBUG("Benchmarking the speed of the bus is done.\n");

	was_benchmarked = 1;
#endif /* !SIMGRID */
}

static void get_bus_path(const char *type, char *path, size_t maxlen)
{
	char hostname[65];
	char *bus;

	bus = _starpu_get_perf_model_dir_bus();
	_starpu_gethostname(hostname, sizeof(hostname));
	snprintf(path, maxlen, "%s%s.%s", bus?_starpu_get_perf_model_dir_bus():"INVALID_LOCATION/", hostname, type);
}

/*
 *	Affinity
 */

static void get_affinity_path(char *path, size_t maxlen)
{
	get_bus_path("affinity", path, maxlen);
}

#ifndef STARPU_SIMGRID

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
	enum starpu_node_kind type;
	unsigned ok = 1;

	for (type = STARPU_CPU_RAM+1; ok && type < STARPU_NRAM; type++)
	{
		for (gpu = 0; ok && gpu < nmem[type]; gpu++)
		{
			int ret;
			unsigned dummy;

			_starpu_drop_comments(f);
			ret = fscanf(f, "%u\t", &dummy);
			if (ret != 1)
			{
				/* Old perfmodel file, ignore rest */
				ok = 0;
				break;
			}

			STARPU_ASSERT(dummy == gpu);

			unsigned numa;
			for (numa = 0; numa < nnumas; numa++)
			{
				ret = fscanf(f, "%u\t", &affinity_matrix[type][gpu][numa]);
				STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);
			}

			ret = fscanf(f, "\n");
			STARPU_ASSERT_MSG(ret == 0, "Error when reading from file '%s'", path);
		}
	}
	if (locked)
		_starpu_frdunlock(f);

	fclose(f);
}

/* NB: we want to sort the bandwidth by DECREASING order */
static int compar_dev_timing(const void *left_dev_timing, const void *right_dev_timing)
{
	const struct dev_timing *left = (const struct dev_timing *)left_dev_timing;
	const struct dev_timing *right = (const struct dev_timing *)right_dev_timing;

	if (left->numa_distance == 0 && right->numa_distance != 0)
		/* We prefer left */
		return -1;

	if (right->numa_distance == 0 && left->numa_distance != 0)
		/* We prefer right */
		return 1;

	if (left->numa_distance >= 0 && right->numa_distance >= 0)
	{
		return left->numa_distance > right->numa_distance ? 1 :
		       left->numa_distance < right->numa_distance ? -1 : 0;
	}

	double left_dtoh = left->timing_dtoh;
	double left_htod = left->timing_htod;
	double right_dtoh = right->timing_dtoh;
	double right_htod = right->timing_htod;

	double timing_sum2_left = left_dtoh*left_dtoh + left_htod*left_htod;
	double timing_sum2_right = right_dtoh*right_dtoh + right_htod*right_htod;

	/* it's for a decreasing sorting */
	return timing_sum2_left > timing_sum2_right ? 1 :
	       timing_sum2_left < timing_sum2_right ? -1 : 0;
}

static void write_bus_affinity_file_content(void)
{
	STARPU_ASSERT(was_benchmarked);

	FILE *f;
	char path[PATH_LENGTH];
	int locked;

	get_affinity_path(path, sizeof(path));

	_STARPU_DEBUG("writing affinities to %s\n", path);

	f = fopen(path, "a+");
	if (!f)
	{
		perror("fopen write_buf_affinity_file_content");
		_STARPU_DISP("path '%s'\n", path);
		fflush(stderr);
		STARPU_ABORT();
	}

	locked = _starpu_fwrlock(f) == 0;
	fseek(f, 0, SEEK_SET);
	_starpu_fftruncate(f, 0);

	unsigned numa;
	unsigned gpu;
	enum starpu_node_kind type;

	fprintf(f, "# GPU\t");
	for (numa = 0; numa < nnumas; numa++)
		fprintf(f, "NUMA%u\t", numa);
	fprintf(f, "\n");

	for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; type++)
	{
		/* Use an other array to sort bandwidth */
		struct dev_timing timing_per_numa_sorted[STARPU_NMAXDEVS][STARPU_MAXNUMANODES];
		memcpy(timing_per_numa_sorted, timing_per_numa[type], sizeof(timing_per_numa[type]));

		for (gpu = 0; gpu < nmem[type]; gpu++)
		{
			fprintf(f, "%u\t", gpu);

			qsort(timing_per_numa_sorted[gpu], nnumas, sizeof(struct dev_timing), compar_dev_timing);

			for (numa = 0; numa < nnumas; numa++)
			{
				fprintf(f, "%d\t", timing_per_numa_sorted[gpu][numa].numa_id);
			}

			fprintf(f, "\n");
		}
	}

	if (locked)
		_starpu_frdunlock(f);
	fclose(f);
}

static void generate_bus_affinity_file(void)
{
	if (!was_benchmarked)
		benchmark_all_memory_nodes();

	write_bus_affinity_file_content();
}

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

static void load_bus_affinity_file(void)
{
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
}

unsigned *_starpu_get_affinity_vector_by_kind(unsigned gpuid, enum starpu_node_kind kind)
{
	return affinity_matrix[kind][gpuid];
}

void starpu_bus_print_affinity(FILE *f)
{
	enum starpu_node_kind type;

	fprintf(f, "# GPU\tNUMA in preference order (logical index)\n");

	for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; type++)
	{
		unsigned gpu;

		if (!nmem[type])
			continue;

		fprintf(f, "# %s\n", starpu_memory_driver_info[type].name_upper);
		for(gpu = 0 ; gpu<nmem[type] ; gpu++)
		{
			unsigned numa;

			fprintf(f, "%u\t", gpu);
			for (numa = 0; numa < nnumas; numa++)
			{
				fprintf(f, "%u\t", affinity_matrix[type][gpu][numa]);
			}
			fprintf(f, "\n");
		}
	}
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
		for (; dst < STARPU_MAXNODES; dst++)
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
	for (; src < STARPU_MAXNODES; src++)
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
			latency_matrix[src][dst] = NAN;

	return 1;
}

#if !defined(STARPU_SIMGRID)
static double search_bus_best_latency(int src, enum starpu_node_kind type, int htod)
{
	/* Search the best latency for this node */
	double best = 0.0;
	double actual = 0.0;
	unsigned check = 0;
	unsigned numa;
	for (numa = 0; numa < nnumas; numa++)
	{
		if (htod)
			actual = timing_per_numa[type][src][numa].latency_htod;
		else
			actual = timing_per_numa[type][src][numa].latency_dtoh;
		if (!check || actual < best)
		{
			best = actual;
			check = 1;
		}
	}
	return best;
}

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

	f = fopen(path, "a+");
	if (!f)
	{
		perror("fopen write_bus_latency_file_content");
		_STARPU_DISP("path '%s'\n", path);
		fflush(stderr);
		STARPU_ABORT();
	}
	locked = _starpu_fwrlock(f) == 0;
	fseek(f, 0, SEEK_SET);
	_starpu_fftruncate(f, 0);

	fprintf(f, "# ");
	for (dst = 0; dst < STARPU_MAXNODES; dst++)
		fprintf(f, "to %u\t\t", dst);
	fprintf(f, "\n");

	enum starpu_node_kind type;
	maxnode = 0;
	for (type = STARPU_CPU_RAM; type < STARPU_NRAM; type++)
		maxnode += nmem[type];

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
				unsigned numa_low = b_low;
				unsigned numa_up = b_up;

				b_low += nnumas;
				/* ---- End NUMA ---- */

				for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; type++)
				{
					b_up += nmem[type];
					/* Check if it's direct GPU-GPU transfer */
					if (src >= b_low && src < b_up && dst >= b_low && dst < b_up
						&& timing_dtod[type][src-b_low][dst-b_low])
						latency += latency_dtod[type][src-b_low][dst-b_low];
					else
					{
						/* Check if it's GPU <-> NUMA link */
						if (src >=b_low && src < b_up && dst >= numa_low && dst < numa_up)
							latency += timing_per_numa[type][(src-b_low)][dst-numa_low].latency_dtoh;
						if (dst >= b_low && dst < b_up && src >= numa_low && dst < numa_up)
							latency += timing_per_numa[type][(dst-b_low)][src-numa_low].latency_htod;
						/* To other devices, take the best latency */
						if (src >= b_low && src < b_up && !(dst >= numa_low && dst < numa_up))
							latency += search_bus_best_latency(src-b_low, type, 0);
						if (dst >= b_low && dst < b_up && !(src >= numa_low && dst < numa_up))
							latency += search_bus_best_latency(dst-b_low, type, 1);
					}
					b_low += nmem[type];
				}
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
		benchmark_all_memory_nodes();

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

			int limit_bandwidth = starpu_getenv_number("STARPU_LIMIT_BANDWIDTH");
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
		for (; dst < STARPU_MAXNODES; dst++)
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
	for (; src < STARPU_MAXNODES; src++)
		for (dst = 0; dst < STARPU_MAXNODES; dst++)
			latency_matrix[src][dst] = NAN;

	return 1;
}

#if !defined(STARPU_SIMGRID)
static double search_bus_best_timing(int src, enum starpu_node_kind type, int htod)
{
	/* Search the best latency for this node */
	double best = 0.0;
	double actual = 0.0;
	unsigned check = 0;
	unsigned numa;
	for (numa = 0; numa < nnumas; numa++)
	{
		if (htod)
			actual = timing_per_numa[type][src][numa].timing_htod;
		else
			actual = timing_per_numa[type][src][numa].timing_dtoh;
		if (!check || actual < best)
		{
			best = actual;
			check = 1;
		}
	}
	return best;
}

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

	f = fopen(path, "a+");
	STARPU_ASSERT_MSG(f, "Error when opening file (writing) '%s'", path);

	locked = _starpu_fwrlock(f) == 0;
	fseek(f, 0, SEEK_SET);
	_starpu_fftruncate(f, 0);

	fprintf(f, "# ");
	for (dst = 0; dst < STARPU_MAXNODES; dst++)
		fprintf(f, "to %u\t\t", dst);
	fprintf(f, "\n");

	enum starpu_node_kind type;
	maxnode = 0;
	for (type = STARPU_CPU_RAM; type < STARPU_NRAM; type++)
		maxnode += nmem[type];

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
				unsigned numa_low = b_low;
				unsigned numa_up = b_up;

				b_low += nnumas;
				/* End NUMA */

				for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; type++)
				{
					b_up += nmem[type];
					/* Check if it's direct GPU-GPU transfer */
					if (src >= b_low && src < b_up && dst >= b_low && dst < b_up
						&& timing_dtod[type][src-b_low][dst-b_low])
						slowness += timing_dtod[type][src-b_low][dst-b_low];
					else
					{
						/* Check if it's GPU <-> NUMA link */
						if (src >= b_low && src < b_up && dst >= numa_low && dst < numa_up)
							slowness += timing_per_numa[type][(src-b_low)][dst-numa_low].timing_dtoh;
						if (dst >= b_low && dst < b_up && src >= numa_low && src < numa_up)
							slowness += timing_per_numa[type][(dst-b_low)][src-numa_low].timing_htod;
						/* To other devices, take the best slowness */
						if (src >= b_low && src < b_up && !(dst >= numa_low && dst < numa_up))
							slowness += search_bus_best_timing(src-b_low, type, 0);
						if (dst >= b_low && dst < b_up && !(src >= numa_low && src < numa_up))
							slowness += search_bus_best_timing(dst-b_low, type, 1);
					}
					b_low += nmem[type];
				}

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

#ifndef STARPU_SIMGRID
	enum starpu_node_kind type;
	int header = 0;
	for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; ++type)
	{
		if (starpu_memory_driver_info[type].ops->calibrate_bus)
		{
			if (!header)
			{
				fprintf(f, "\nGPU\tNUMA in preference order (logical index), host-to-device, device-to-host\n");
				header = 1;
			}
			enum starpu_worker_archtype arch = starpu_memory_node_get_worker_archtype(type);
			const char *type_str = starpu_worker_get_type_as_string(arch);
			for (src = 0; src < nmem[type]; src++)
			{
				struct dev_timing *timing;
				struct _starpu_machine_config * config = _starpu_get_machine_config();
				unsigned nhwnumas = _starpu_topology_get_nhwnumanodes(config);
				unsigned numa;
				fprintf(f, "%s%u\t", type_str, src);
				for (numa = 0; numa < nhwnumas; numa++)
				{
					timing = &timing_per_numa[type][src][numa];
					if (timing->timing_htod)
						fprintf(f, "%2d %.0f %.0f\t", timing->numa_id, 1/timing->timing_htod, 1/timing->timing_dtoh);
					else
						fprintf(f, "%2u\t", affinity_matrix[type][src][numa]);
				}
				fprintf(f, "\n");
			}
		}
	}
#endif
}

static void generate_bus_bandwidth_file(void)
{
	if (!was_benchmarked)
		benchmark_all_memory_nodes();

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

static void compare_value_and_recalibrate(enum starpu_node_kind type, const char * msg, unsigned val_file, unsigned val_detected)
{
	int recalibrate = 0;
	if (val_file != val_detected &&
		!((type == STARPU_MPI_MS_RAM || type == STARPU_TCPIP_MS_RAM) && !val_detected))
		recalibrate = 1;

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	//Send to each other to know if we had to recalibrate because someone cannot have the correct value in the config file
	if (_starpu_config.conf.nmpi_ms != 0)
		recalibrate = mpi_check_recalibrate(recalibrate);
#endif

	if (recalibrate)
	{
#ifdef STARPU_USE_MPI_MASTER_SLAVE
		/* Only the master prints the message */
		if (_starpu_mpi_common_is_src_node())
#endif
			_STARPU_DISP("Current configuration does not match the bus performance model (%s: (stored) %d != (current) %d), recalibrating...\n", msg, val_file, val_detected);

		int location = _starpu_get_perf_model_bus();
		_starpu_bus_force_sampling(location);

#ifdef STARPU_USE_MPI_MASTER_SLAVE
		if (_starpu_mpi_common_is_src_node())
#endif
			_STARPU_DISP("... done\n");
	}
}

static void check_bus_config_file(void)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	int recalibrate = 0;
	char path[PATH_LENGTH];

	int location = _starpu_get_perf_model_bus();
	if (location < 0 || config->conf.bus_calibrate > 0)
		recalibrate = 1;

#if defined(STARPU_USE_MPI_MASTER_SLAVE)
	if (_starpu_config.conf.nmpi_ms != 0)
		//Send to each other to know if we had to recalibrate because someone cannot have the config file
		recalibrate = mpi_check_recalibrate(recalibrate);
#endif

	if (recalibrate)
	{
		if (location < 0)
			_STARPU_DISP("No performance model for the bus, calibrating...\n");
		_starpu_bus_force_sampling(location);
		if (location < 0)
			_STARPU_DISP("... done\n");
	}
	else
	{
		FILE *f;
		int ret;
		enum starpu_node_kind type;
		unsigned read_cpus = -1;
		unsigned n_read[STARPU_NRAM];

		int locked;
		unsigned ok;

		get_config_path(path, sizeof(path));

		// Loading configuration from file
		f = fopen(path, "r");
		STARPU_ASSERT_MSG(f, "Error when reading from file '%s'", path);
		locked = _starpu_frdlock(f) == 0;
		_starpu_drop_comments(f);

		ret = fscanf(f, "%u\t", &read_cpus);
		STARPU_ASSERT_MSG(ret == 1, "Error when reading from file '%s'", path);
		_starpu_drop_comments(f);

		for (type = STARPU_CPU_RAM; type < STARPU_NRAM; type++)
			n_read[type] = -1;
		ok = 1;
		for (type = STARPU_CPU_RAM; ok && type < STARPU_NRAM; type++)
		{
			if (ok)
				ret = fscanf(f, "%u\t", &n_read[type]);
			if (!ok || ret != 1)
			{
				ok = 0;
				n_read[type] = 0;
			}
			_starpu_drop_comments(f);
		}

		if (locked)
			_starpu_frdunlock(f);
		fclose(f);

		// Loading current configuration
		ncpus = _starpu_topology_get_nhwcpu(config);
		nnumas = _starpu_topology_get_nhwnumanodes(config);

		enum starpu_worker_archtype arch;
		for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; ++type)
		{
			if (type != STARPU_DISK_RAM)
			{
				arch = starpu_memory_node_get_worker_archtype(type);
				nmem[type] = topology->nhwdevices[arch];
			}
		}

		// Checking if both configurations match
		compare_value_and_recalibrate(STARPU_CPU_RAM, "CPUS", read_cpus, ncpus);
		for (type = STARPU_CPU_RAM; type < STARPU_NRAM; type++)
		{
			compare_value_and_recalibrate(type,
				starpu_memory_driver_info[type].name_upper, n_read[type], nmem[type]);
		}
	}
}

static void write_bus_config_file_content(void)
{
	FILE *f;
	char path[PATH_LENGTH];
	int locked;
	enum starpu_node_kind type;

	STARPU_ASSERT(was_benchmarked);
	get_config_path(path, sizeof(path));

	_STARPU_DEBUG("writing config to %s\n", path);

	f = fopen(path, "a+");
	STARPU_ASSERT_MSG(f, "Error when opening file (writing) '%s'", path);
	locked = _starpu_fwrlock(f) == 0;
	fseek(f, 0, SEEK_SET);
	_starpu_fftruncate(f, 0);

	fprintf(f, "# Current configuration\n");
	fprintf(f, "%u # Number of CPUs\n", ncpus);
	for (type = STARPU_CPU_RAM; type < STARPU_NRAM; type++)
		fprintf(f, "%u # Number of %s nodes\n", nmem[type],
				starpu_memory_driver_info[type].name_upper);

	if (locked)
		_starpu_fwrunlock(f);
	fclose(f);
}

static void generate_bus_config_file(void)
{
	if (!was_benchmarked)
		benchmark_all_memory_nodes();

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

#if defined(STARPU_USE_CUDA) && HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX && defined(STARPU_HAVE_CUDA_MEMCPY_PEER)

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

/* Our traffic had to go through the host, go back from target up to the host,
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
		/* We have to go up to the Interconnect, so obj2 is not in the same PCI
		 * tree, so we're for for obj1 to Interconnect, and just find the path
		 * from obj2 to Interconnect too.
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

static hwloc_obj_t get_hwloc_cuda_obj(hwloc_topology_t topology, unsigned devid)
{
	hwloc_obj_t res;
	struct cudaDeviceProp props;
	cudaError_t cures;

	res = hwloc_cuda_get_device_osdev_by_index(topology, devid);
	if (res)
		return res;

	cures = cudaGetDeviceProperties(&props, devid);
	if (cures == cudaSuccess)
	{
		res = hwloc_get_pcidev_by_busid(topology, props.pciDomainID, props.pciBusID, props.pciDeviceID, 0);
		if (res)
			return res;

#if defined(STARPU_HAVE_LIBNVIDIA_ML) && !defined(STARPU_USE_CUDA0) && !defined(STARPU_USE_CUDA1)
		nvmlDevice_t nvmldev = _starpu_cuda_get_nvmldev(&props);

		if (nvmldev)
		{
			unsigned int index;
			if (nvmlDeviceGetIndex(nvmldev, &index) == NVML_SUCCESS)
			{
				res = hwloc_nvml_get_device_osdev_by_index(topology, index);
				if (res)
					return res;
			}

			res = hwloc_nvml_get_device_osdev(topology, nvmldev);
			if (res)
				return res;
		}
#endif
	}
	return NULL;
}

/* find the path between cuda i and cuda j, and update the maximum bandwidth along the path */
static int find_platform_cuda_path(hwloc_topology_t topology, unsigned i, unsigned j, double bandwidth)
{
	hwloc_obj_t cudai, cudaj;
	cudai = get_hwloc_cuda_obj(topology, i);
	cudaj = get_hwloc_cuda_obj(topology, j);

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
	/* We don't care about traffic going through PCI switches */
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

/* Our traffic has to go through the host, go back from target up to the host,
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
		/* Finished, go through NUMA */
		hwloc_obj_t numa = _starpu_numa_get_obj(obj);
		if (numa)
			fprintf(f, "    <link_ctn id=\"NUMA%d\"/>\n", numa->logical_index);
		else
			fprintf(f, "     <link_ctn id=\"Interconnect\"/>\n");
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
		/* Finished, go through NUMA */
		hwloc_obj_t numa = _starpu_numa_get_obj(obj);
		if (numa)
			fprintf(f, "    <link_ctn id=\"NUMA%d\"/>\n", numa->logical_index);
		else
			fprintf(f, "     <link_ctn id=\"Interconnect\"/>\n");
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
		/* We have to go up to the Interconnect, so obj2 is not in the same PCI
		 * tree, so we're for for obj1 to Interconnect, and just find the path
		 * from obj2 to Interconnect too.
		 */
		emit_platform_backward_path(f, obj2);

		hwloc_obj_t numa2 = _starpu_numa_get_obj(obj2);
		hwloc_obj_t numa1 = _starpu_numa_get_obj(obj1);

		if (!numa1 || !numa2 || numa1 != numa2)
		{
			fprintf(f, "    <link_ctn id=\"Interconnect\"/>\n");
			if (numa1)
				fprintf(f, "    <link_ctn id=\"NUMA%d\"/>\n", numa1->logical_index);
		}

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

static int have_hwloc_pci_bw_routes(FILE *f, const char *Bps, const char *s);
static void write_bus_platform_file_content(int version)
{
	FILE *f;
	char path[PATH_LENGTH];
	unsigned i;
	const char *speed, *flops, *Bps, *s;
	char dash;
	int locked;
	enum starpu_node_kind type;

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

	f = fopen(path, "a+");
	if (!f)
	{
		perror("fopen write_bus_platform_file_content");
		_STARPU_DISP("path '%s'\n", path);
		fflush(stderr);
		STARPU_ABORT();
	}
	locked = _starpu_fwrlock(f) == 0;
	fseek(f, 0, SEEK_SET);
	_starpu_fftruncate(f, 0);

	fprintf(f,
			"<?xml version='1.0'?>\n"
			"<!DOCTYPE platform SYSTEM '%s'>\n"
			" <platform version=\"%d\">\n"
			" <config id=\"General\">\n"
			"   <prop id=\"network/TCP%cgamma\" value=\"-1\"></prop>\n"
			"   <prop id=\"network/latency%cfactor\" value=\"1\"></prop>\n"
			"   <prop id=\"network/bandwidth%cfactor\" value=\"1\"></prop>\n"
			"   <prop id=\"network/crosstraffic\" value=\"0\"></prop>\n"
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

	for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; ++type)
	{
		if (starpu_memory_driver_info[type].ops->calibrate_bus)
		{
			for (i = 0; i < nmem[type]; i++)
			{
				fprintf(f, "   <host id=\"%s%u\" %s=\"2000000000%s\">\n",
					starpu_memory_driver_info[type].name_upper, i, speed, flops);
				fprintf(f, "     <prop id=\"model\" value=\"%s\"/>\n", device_name[type][i]);
				fprintf(f, "     <prop id=\"memsize\" value=\"%llu\"/>\n",
					(unsigned long long) device_memory[type][i]);
				/* TODO: record device_peer_access to properly express which gpu-gpu direct connections exist. */
				fprintf(f, "     <prop id=\"memcpy_peer\" value=\"%d\"/>\n", 1);
				fprintf(f, "   </host>\n");
			}
		}
	}
	fprintf(f, "\n   <host id=\"RAM\" %s=\"1%s\"/>\n", speed, flops);

	/*
	 * Compute maximum bandwidth, taken as host bandwidth
	 */
	double max_bandwidth = 0;
	double max_bandwidth_numa[nnumas];
	unsigned numa;
	for (numa = 0; numa < nnumas; numa++)
		max_bandwidth_numa[numa] = 0.;

	for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; ++type)
	{
		if (starpu_memory_driver_info[type].ops->calibrate_bus)
		{
			for (i = 0; i < nmem[type]; i++)
			{
				for (numa = 0; numa < nnumas; numa++)
				{
					double down_bw = 1.0 / timing_per_numa[type][i][numa].timing_dtoh;
					double up_bw = 1.0 / timing_per_numa[type][i][numa].timing_htod;
					if (max_bandwidth < down_bw)
						max_bandwidth = down_bw;
					if (max_bandwidth_numa[numa] < down_bw)
						max_bandwidth_numa[numa] = down_bw;
					if (max_bandwidth < up_bw)
						max_bandwidth = up_bw;
					if (max_bandwidth_numa[numa] < up_bw)
						max_bandwidth_numa[numa] = up_bw;
				}
			}
		}
	}

	for (numa = 0; numa < nnumas; numa++)
		fprintf(f, "   <link id=\"NUMA%d\" bandwidth=\"%f%s\" latency=\"0.000000%s\"/>\n", numa, max_bandwidth_numa[numa]*1000000, Bps, s);
	fprintf(f, "   <link id=\"Interconnect\" bandwidth=\"%f%s\" latency=\"0.000000%s\"/>\n\n", max_bandwidth*1000000, Bps, s);

	/*
	 * Device links and routes
	 */

	for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; ++type)
	{
		if (starpu_memory_driver_info[type].ops->calibrate_bus)
		{
			const char *name = starpu_memory_driver_info[type].name_upper;
			/* Write RAM/Device bandwidths and latencies */
			for (i = 0; i < nmem[type]; i++)
			{
				char i_name[16];
				snprintf(i_name, sizeof(i_name), "%s%u", name, i);
				fprintf(f, "   <link id=\"RAM-%s\" bandwidth=\"%f%s\" latency=\"%f%s\"/>\n",
					i_name,
					1000000. / search_bus_best_timing(i, type, 1), Bps,
					search_bus_best_latency(i, type, 1)/1000000., s);
				fprintf(f, "   <link id=\"%s-RAM\" bandwidth=\"%f%s\" latency=\"%f%s\"/>\n",
					i_name,
					1000000. / search_bus_best_timing(i, type, 0), Bps,
					search_bus_best_latency(i, type, 0)/1000000., s);
			}
			fprintf(f, "\n");
			/* Write Device/Device bandwidths and latencies */
			for (i = 0; i < nmem[type]; i++)
			{
				unsigned j;
				char i_name[16];
				snprintf(i_name, sizeof(i_name), "%s%u", name, i);
				for (j = 0; j < nmem[type]; j++)
				{
					char j_name[16];
					if (j == i || !device_peer_access[type][i][j])
						continue;
					snprintf(j_name, sizeof(j_name), "%s%u", name, j);
					fprintf(f, "   <link id=\"%s-%s\" bandwidth=\"%f%s\" latency=\"%f%s\"/>\n",
						i_name, j_name,
						1000000. / timing_dtod[type][i][j], Bps,
						latency_dtod[type][i][j]/1000000., s);
				}
			}

			if (!have_hwloc_pci_bw_routes(f, Bps, s))
			{
				/* If we don't have enough hwloc information, write trivial routes always through host */
				for (i = 0; i < nmem[type]; i++)
				{
					char i_name[16];
					snprintf(i_name, sizeof(i_name), "%s%u", name, i);
					fprintf(f, "   <route src=\"RAM\" dst=\"%s\" symmetrical=\"NO\">\n", i_name);
					fprintf(f, "      <link_ctn id=\"RAM-%s\"/>\n", i_name);
					fprintf(f, "      <link_ctn id=\"Interconnect\"/>\n");
					fprintf(f, "   </route>\n");
					fprintf(f, "   <route src=\"%s\" dst=\"RAM\" symmetrical=\"NO\">\n", i_name);
					fprintf(f, "      <link_ctn id=\"%s-RAM\"/>\n", i_name);
					fprintf(f, "      <link_ctn id=\"Interconnect\"/>\n");
					fprintf(f, "   </route>\n");
				}
				for (i = 0; i < nmem[type]; i++)
				{
					unsigned j;
					char i_name[16];
					snprintf(i_name, sizeof(i_name), "%s%u", name, i);
					for (j = 0; j < nmem[type]; j++)
					{
						char j_name[16];
						if (j == i || !device_peer_access[type][i][j])
							continue;
						snprintf(j_name, sizeof(j_name), "%s%u", name, j);
						fprintf(f, "   <route src=\"%s\" dst=\"%s\" symmetrical=\"NO\">\n", i_name, j_name);
						fprintf(f, "     <link_ctn id=\"%s-%s\"/>\n", i_name, j_name);
						fprintf(f, "     <link_ctn id=\"Interconnect\"/>\n");
						fprintf(f, "   </route>\n");
					}
				}
			}
			fprintf(f, "\n");
		}
	}

	fprintf(f,
		" </AS>\n"
		" </platform>\n"
	       );

	if (locked)
		_starpu_fwrunlock(f);
	fclose(f);

}

static int have_hwloc_pci_bw_routes(FILE *f, const char *Bps, const char *s)
{
	(void)f;
	(void)Bps;
	(void)s;
#if HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX && defined(STARPU_USE_CUDA) && defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
	unsigned i;

	/* If we have enough hwloc information, write PCI bandwidths and routes */
	if (!starpu_getenv_number_default("STARPU_PCI_FLAT", 0) && ncuda > 0)
	{
		int ret;
		hwloc_topology_t topology;
		ret = hwloc_topology_init(&topology);
		STARPU_ASSERT_MSG(ret == 0, "Could not initialize Hwloc topology (%s)\n", strerror(errno));
		_starpu_topology_filter(topology);
		ret = hwloc_topology_load(topology);
		STARPU_ASSERT_MSG(ret == 0, "Could not load Hwloc topology (%s)\n", strerror(errno));

		char nvlink[ncuda][ncuda];
		char nvlinkhost[ncuda];
		char nvswitch[ncuda];
		memset(nvlink, 0, sizeof(nvlink));
		memset(nvlinkhost, 0, sizeof(nvlinkhost));
		memset(nvswitch, 0, sizeof(nvswitch));

		/* TODO: move to drivers */
#if defined(STARPU_HAVE_LIBNVIDIA_ML) && !defined(STARPU_USE_CUDA0) && !defined(STARPU_USE_CUDA1)
		/* First find NVLinks */
		struct cudaDeviceProp props[ncuda];

		for (i = 0; i < ncuda; i++)
		{
			cudaError_t cures = cudaGetDeviceProperties(&props[i], i);
			if (cures != cudaSuccess)
				props[i].name[0] = 0;
		}

		for (i = 0; i < ncuda; i++)
		{
			unsigned j;

			if (!props[i].name[0])
				continue;

			nvmlDevice_t nvmldev;
			nvmldev = _starpu_cuda_get_nvmldev(&props[i]);
			if (!nvmldev)
				continue;

			for (j = 0; j < NVML_NVLINK_MAX_LINKS; j++)
			{
				nvmlEnableState_t active;
				nvmlReturn_t nvmlret;
				nvmlPciInfo_t pci;
				unsigned k;

				nvmlret = nvmlDeviceGetNvLinkState(nvmldev, j, &active);
				if (nvmlret != NVML_SUCCESS)
					continue;
				if (active != NVML_FEATURE_ENABLED)
					continue;
				nvmlret = nvmlDeviceGetNvLinkRemotePciInfo(nvmldev, j, &pci);
				if (nvmlret != NVML_SUCCESS)
					continue;

				hwloc_obj_t obj = hwloc_get_pcidev_by_busid(topology,
					        pci.domain, pci.bus, pci.device, 0);
				if (obj && obj->type == HWLOC_OBJ_PCI_DEVICE && (obj->attr->pcidev.class_id >> 8 == 0x06))
				{
					/* This is a PCI bridge */
					switch (obj->attr->pcidev.vendor_id)
					{
					case 0x1014:
						/* IBM OpenCAPI port, direct CPU-GPU NVLink */
						/* TODO: NUMA affinity */
						nvlinkhost[i] = 1;
						continue;
					case 0x10de:
						nvswitch[i] = 1;
						continue;
					}
				}

				/* Otherwise, link to another GPU? */
				for (k = 0; k < ncuda; k++)
				{
					if ((int) pci.domain == props[k].pciDomainID
					 && (int) pci.bus == props[k].pciBusID
					 && (int) pci.device == props[k].pciDeviceID)
					{
						nvlink[i][k] = 1;
						nvlink[k][i] = 1;
						break;
					}
				}
				if (k < ncuda)
					/* Yes it was another GPU */
					continue;

				/* No idea what this is */
				_STARPU_DISP("Warning: NVLink to unknown PCI card %04x:%02x:%02x: %04x\n", pci.domain, pci.bus, pci.device, pci.pciDeviceId);
			}
		}

		for (i = 0; i < ncuda; i++)
		{
			unsigned j;
			for (j = i+1; j < ncuda; j++)
			{
				if (nvswitch[i] && nvswitch[j])
				{
					static int warned = 0;
					if (!warned)
					{
						warned = 1;
						/* TODO: follow answers to https://forums.developer.nvidia.com/t/how-to-distinguish-different-nvswitch/241983 */
						_STARPU_DISP("Warning: NVSwitch not tested yet with several switches, assuming there is only one NVSwitch in the system\n");
					}
					nvlink[i][j] = 1;
					nvlink[j][i] = 1;
				}
			}
		}
#endif

		/* Find paths and record measured bandwidth along the path */
		for (i = 0; i < ncuda; i++)
		{
			unsigned j;

			for (j = 0; j < ncuda; j++)
				if (i != j && !nvlink[i][j] && !nvlinkhost[i] && !nvlinkhost[j])
					if (!find_platform_cuda_path(topology, i, j, 1000000. / timing_dtod[STARPU_CUDA_RAM][i][j]))
					{
						_STARPU_DISP("Warning: could not get CUDA location from hwloc\n");
						clean_topology(hwloc_get_root_obj(topology));
						hwloc_topology_destroy(topology);
						return 0;
					}

			/* Record RAM/CUDA bandwidths */
			if (!nvlinkhost[i])
			{
				find_platform_forward_path(get_hwloc_cuda_obj(topology, i), 1000000. / search_bus_best_timing(i, STARPU_CUDA_RAM, 0));
				find_platform_backward_path(get_hwloc_cuda_obj(topology, i), 1000000. / search_bus_best_timing(i, STARPU_CUDA_RAM, 1));
			}
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
					if (!nvlink[i][j])
					{
						if (nvlinkhost[i] && nvlinkhost[j])
						{
							/* FIXME: if they are directly connected through PCI, is NVLink host preferred? */
							if (gpu_numa[STARPU_CUDA_RAM][i] >= 0)
								fprintf(f, "    <link_ctn id=\"NUMA%d\"/>\n", gpu_numa[STARPU_CUDA_RAM][i]);
							fprintf(f, "    <link_ctn id=\"Interconnect\"/>\n");
							if (gpu_numa[STARPU_CUDA_RAM][j] >= 0)
								fprintf(f, "    <link_ctn id=\"NUMA%d\"/>\n", gpu_numa[STARPU_CUDA_RAM][j]);
						}
						else
							emit_platform_path_up(f,
									get_hwloc_cuda_obj(topology, i),
									get_hwloc_cuda_obj(topology, j));
					}
					fprintf(f, "   </route>\n");
				}

			fprintf(f, "   <route src=\"CUDA%u\" dst=\"RAM\" symmetrical=\"NO\">\n", i);
			fprintf(f, "    <link_ctn id=\"CUDA%u-RAM\"/>\n", i);
			if (nvlinkhost[i])
			{
				if (gpu_numa[STARPU_CUDA_RAM][i] >= 0)
					fprintf(f, "    <link_ctn id=\"NUMA%d\"/>\n", gpu_numa[STARPU_CUDA_RAM][i]);
			}
			else
				emit_platform_forward_path(f, get_hwloc_cuda_obj(topology, i));
			fprintf(f, "   </route>\n");

			fprintf(f, "   <route src=\"RAM\" dst=\"CUDA%u\" symmetrical=\"NO\">\n", i);
			fprintf(f, "    <link_ctn id=\"RAM-CUDA%u\"/>\n", i);
			if (nvlinkhost[i])
			{
				if (gpu_numa[STARPU_CUDA_RAM][i] >= 0)
					fprintf(f, "    <link_ctn id=\"NUMA%d\"/>\n", gpu_numa[STARPU_CUDA_RAM][i]);
			}
			else
				emit_platform_backward_path(f, get_hwloc_cuda_obj(topology, i));
			fprintf(f, "   </route>\n");
		}

		clean_topology(hwloc_get_root_obj(topology));
		hwloc_topology_destroy(topology);
		return 1;
	}
#endif /* HAVE_DECL_HWLOC_CUDA_GET_DEVICE_OSDEV_BY_INDEX */
	return 0;
}

static void generate_bus_platform_file(void)
{
	if (!was_benchmarked)
		benchmark_all_memory_nodes();

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

static void _starpu_bus_force_sampling(int location)
{
	_STARPU_DEBUG("Force bus sampling ...\n");
	if (location < 0)
	{
		location = _starpu_set_default_perf_model_bus();
	}
	_starpu_create_bus_sampling_directory_if_needed(location);

	generate_bus_affinity_file();
	generate_bus_latency_file();
	generate_bus_bandwidth_file();
	generate_bus_config_file();
	generate_bus_platform_file();
}
#endif /* !SIMGRID */

void _starpu_load_bus_performance_files(void)
{
	_starpu_create_bus_sampling_directory_if_needed(-1);

	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &_starpu_get_machine_config()->topology;
	nnumas = _starpu_topology_get_nhwnumanodes(config);
#ifndef STARPU_SIMGRID
	ncpus = _starpu_topology_get_nhwcpu(config);
#endif

	enum starpu_worker_archtype arch;
	enum starpu_node_kind type;
	for (type = STARPU_CPU_RAM+1; type < STARPU_NRAM; ++type)
	{
		if (type != STARPU_DISK_RAM)
		{
			arch = starpu_memory_node_get_worker_archtype(type);
			nmem[type] = topology->nhwdevices[arch];
		}
	}

#ifndef STARPU_SIMGRID
	check_bus_config_file();
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	/* be sure that master wrote the perf files */
	if (_starpu_config.conf.nmpi_ms != 0)
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
	int busid = starpu_bus_get_id(src_node, dst_node);
#if 0
	int direct = starpu_bus_get_direct(busid);
#endif
	float ngpus = starpu_bus_get_ngpus(busid);
	if (ngpus != 1)
		ngpus = topology->ndevices[STARPU_CUDA_WORKER]
			+topology->ndevices[STARPU_HIP_WORKER]
			+topology->ndevices[STARPU_OPENCL_WORKER];
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
	int print_stats = starpu_getenv_number_default("STARPU_BUS_STATS", 0);

	if (print_stats)
	{
		fprintf(stderr, "\n#---------------------\n");
		fprintf(stderr, "Data transfer speed for %s (node %u):\n", name, node);
	}

	/* save bandwidth */
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
