/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
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

#include <math.h>
#include <starpu.h>
#include <starpu_profiling.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/debug.h>
#include <starpu_opencl.h>
#include <drivers/driver_common/driver_common.h>
#include <drivers/opencl/driver_opencl.h>
#include <drivers/opencl/driver_opencl_utils.h>
#include <common/utils.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <core/task.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

static int nb_devices = -1;
static int init_done = 0;

static starpu_pthread_mutex_t big_lock = STARPU_PTHREAD_MUTEX_INITIALIZER;

static size_t global_mem[STARPU_MAXOPENCLDEVS];

#ifdef STARPU_USE_OPENCL
static cl_context contexts[STARPU_MAXOPENCLDEVS];
static cl_device_id devices[STARPU_MAXOPENCLDEVS];
static cl_command_queue queues[STARPU_MAXOPENCLDEVS];
static cl_command_queue in_transfer_queues[STARPU_MAXOPENCLDEVS];
static cl_command_queue out_transfer_queues[STARPU_MAXOPENCLDEVS];
static cl_command_queue peer_transfer_queues[STARPU_MAXOPENCLDEVS];
#ifndef STARPU_SIMGRID
static cl_command_queue alloc_queues[STARPU_MAXOPENCLDEVS];
static cl_event task_events[STARPU_MAXOPENCLDEVS][STARPU_MAX_PIPELINE];
#endif /* !STARPU_SIMGRID */
#endif
#ifdef STARPU_SIMGRID
static unsigned task_finished[STARPU_MAXOPENCLDEVS][STARPU_MAX_PIPELINE];
static starpu_pthread_mutex_t opencl_alloc_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
#endif /* STARPU_SIMGRID */

#define _STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err) do { if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err); } while(0)

void
_starpu_opencl_discover_devices(struct _starpu_machine_config *config)
{
	/* Discover the number of OpenCL devices. Fill the result in CONFIG. */
	/* As OpenCL must have been initialized before calling this function,
	 * `nb_device' is ensured to be correctly set. */
	STARPU_ASSERT(init_done == 1);
	config->topology.nhwopenclgpus = nb_devices;
}

static void _starpu_opencl_limit_gpu_mem_if_needed(unsigned devid)
{
	starpu_ssize_t limit;
	size_t STARPU_ATTRIBUTE_UNUSED totalGlobalMem = 0;
	size_t STARPU_ATTRIBUTE_UNUSED to_waste = 0;

#ifdef STARPU_SIMGRID
	totalGlobalMem = _starpu_simgrid_get_memsize("OpenCL", devid);
#elif defined(STARPU_USE_OPENCL)
	/* Request the size of the current device's memory */
	cl_int err;
	cl_ulong size;
	err = clGetDeviceInfo(devices[devid], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size), &size, NULL);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
	totalGlobalMem = size;
#endif

	limit = starpu_get_env_number("STARPU_LIMIT_OPENCL_MEM");
	if (limit == -1)
	{
		char name[30];
		snprintf(name, sizeof(name), "STARPU_LIMIT_OPENCL_%u_MEM", devid);
		limit = starpu_get_env_number(name);
	}
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (limit == -1)
	{
		/* Use 90% of the available memory by default.  */
		limit = totalGlobalMem / (1024*1024) * 0.9;
	}
#endif

	global_mem[devid] = limit * 1024*1024;

#ifdef STARPU_USE_OPENCL
	/* How much memory to waste ? */
	to_waste = totalGlobalMem - global_mem[devid];
#endif

	_STARPU_DEBUG("OpenCL device %u: Wasting %ld MB / Limit %ld MB / Total %ld MB / Remains %ld MB\n",
			devid, (long)to_waste/(1024*1024), (long) limit, (long)totalGlobalMem/(1024*1024),
			(long)(totalGlobalMem - to_waste)/(1024*1024));

}

#ifdef STARPU_USE_OPENCL
void starpu_opencl_get_context(int devid, cl_context *context)
{
        *context = contexts[devid];
}

void starpu_opencl_get_device(int devid, cl_device_id *device)
{
        *device = devices[devid];
}

void starpu_opencl_get_queue(int devid, cl_command_queue *queue)
{
        *queue = queues[devid];
}

void starpu_opencl_get_current_queue(cl_command_queue *queue)
{
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	STARPU_ASSERT(queue);
        *queue = queues[worker->devid];
}

void starpu_opencl_get_current_context(cl_context *context)
{
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	STARPU_ASSERT(context);
        *context = contexts[worker->devid];
}
#endif /* STARPU_USE_OPENCL */

int _starpu_opencl_init_context(int devid)
{
#ifdef STARPU_SIMGRID
	int j;
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
		task_finished[devid][j] = 0;
#else /* !STARPU_SIMGRID */
	cl_int err;
	cl_uint uint;

	STARPU_PTHREAD_MUTEX_LOCK(&big_lock);

        _STARPU_DEBUG("Initialising context for dev %d\n", devid);

        // Create a compute context
	err = 0;
        contexts[devid] = clCreateContext(NULL, 1, &devices[devid], NULL, NULL, &err);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clGetDeviceInfo(devices[devid], CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(uint), &uint, NULL);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
	starpu_malloc_set_align(uint/8);

        // Create execution queue for the given device
        queues[devid] = clCreateCommandQueue(contexts[devid], devices[devid], 0, &err);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        // Create transfer queue for the given device
        cl_command_queue_properties props;
        err = clGetDeviceInfo(devices[devid], CL_DEVICE_QUEUE_PROPERTIES, sizeof(props), &props, NULL);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        props &= ~CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        in_transfer_queues[devid] = clCreateCommandQueue(contexts[devid], devices[devid], props, &err);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        out_transfer_queues[devid] = clCreateCommandQueue(contexts[devid], devices[devid], props, &err);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        peer_transfer_queues[devid] = clCreateCommandQueue(contexts[devid], devices[devid], props, &err);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        alloc_queues[devid] = clCreateCommandQueue(contexts[devid], devices[devid], 0, &err);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

	STARPU_PTHREAD_MUTEX_UNLOCK(&big_lock);
#endif /* !STARPU_SIMGRID */
	return 0;
}

int _starpu_opencl_deinit_context(int devid)
{
#ifdef STARPU_SIMGRID
	int j;
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
		task_finished[devid][j] = 0;
#else /* !STARPU_SIMGRID */
        cl_int err;

	STARPU_PTHREAD_MUTEX_LOCK(&big_lock);

        _STARPU_DEBUG("De-initialising context for dev %d\n", devid);

        err = clFinish(queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clReleaseCommandQueue(queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clFinish(in_transfer_queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clReleaseCommandQueue(in_transfer_queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clFinish(out_transfer_queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clReleaseCommandQueue(out_transfer_queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clFinish(peer_transfer_queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clReleaseCommandQueue(peer_transfer_queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clFinish(alloc_queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clReleaseCommandQueue(alloc_queues[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        err = clReleaseContext(contexts[devid]);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        contexts[devid] = NULL;

	STARPU_PTHREAD_MUTEX_UNLOCK(&big_lock);
#endif

        return 0;
}

#ifdef STARPU_USE_OPENCL
cl_int starpu_opencl_allocate_memory(int devid STARPU_ATTRIBUTE_UNUSED, cl_mem *mem STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED, cl_mem_flags flags STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	STARPU_ABORT();
#else
	cl_int err;
        cl_mem memory;

	memory = clCreateBuffer(contexts[devid], flags, size, NULL, &err);
	if (err == CL_OUT_OF_HOST_MEMORY)
		return err;
	if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
		return err;
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

	/*
	 * OpenCL uses lazy memory allocation: we will only know if the
	 * allocation failed when trying to copy data onto the device. But we
	 * want to know this __now__, so we just perform a dummy copy.
	 */
	char dummy = 0;
	cl_event ev;
	err = clEnqueueWriteBuffer(alloc_queues[devid], memory, CL_TRUE,
				   0, sizeof(dummy), &dummy,
				   0, NULL, &ev);
	if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
		return err;
	if (err == CL_OUT_OF_RESOURCES)
		return err;
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

	clWaitForEvents(1, &ev);
	clReleaseEvent(ev);

        *mem = memory;
        return CL_SUCCESS;
#endif
}

cl_int starpu_opencl_copy_ram_to_opencl(void *ptr, unsigned src_node STARPU_ATTRIBUTE_UNUSED, cl_mem buffer, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size, size_t offset, cl_event *event, int *ret)
{
	cl_int err;
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	double start = 0.;

	if (event)
		starpu_interface_start_driver_copy_async(src_node, dst_node, &start);

	cl_event ev;
	err = clEnqueueWriteBuffer(in_transfer_queues[worker->devid], buffer, CL_FALSE, offset, size, ptr, 0, NULL, &ev);

	if (event)
		starpu_interface_end_driver_copy_async(src_node, dst_node, start);

	if (STARPU_LIKELY(err == CL_SUCCESS))
	{
		if (event == NULL)
		{
			/* We want a synchronous copy, let's synchronise the queue */
			err = clWaitForEvents(1, &ev);
			_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

			err = clReleaseEvent(ev);
			_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
		}
		else
		{
			clFlush(in_transfer_queues[worker->devid]);
			*event = ev;
		}

		if (ret)
		{
			*ret = (event == NULL) ? 0 : -EAGAIN;
		}
	}
	return err;
}

cl_int starpu_opencl_copy_opencl_to_ram(cl_mem buffer, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *ptr, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size, size_t offset, cl_event *event, int *ret)
{
	cl_int err;
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	double start = 0.;

	if (event)
		starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
	cl_event ev;
	err = clEnqueueReadBuffer(out_transfer_queues[worker->devid], buffer, CL_FALSE, offset, size, ptr, 0, NULL, &ev);
	if (event)
		starpu_interface_end_driver_copy_async(src_node, dst_node, start);
	if (STARPU_LIKELY(err == CL_SUCCESS))
	{
		if (event == NULL)
		{
			/* We want a synchronous copy, let's synchronise the queue */
			err = clWaitForEvents(1, &ev);
			_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

			err = clReleaseEvent(ev);
			_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
		}
		else
		{
			clFlush(out_transfer_queues[worker->devid]);
			*event = ev;
		}

		if (ret)
		{
			*ret = (event == NULL) ? 0 : -EAGAIN;
		}
	}
	return err;
}

cl_int starpu_opencl_copy_opencl_to_opencl(cl_mem src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, size_t src_offset, cl_mem dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t dst_offset, size_t size, cl_event *event, int *ret)
{
	cl_int err;
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	double start = 0.;

	if (event)
		starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
	cl_event ev;
	err = clEnqueueCopyBuffer(peer_transfer_queues[worker->devid], src, dst, src_offset, dst_offset, size, 0, NULL, &ev);
	if (event)
		starpu_interface_end_driver_copy_async(src_node, dst_node, start);
	if (STARPU_LIKELY(err == CL_SUCCESS))
	{
		if (event == NULL)
		{
			/* We want a synchronous copy, let's synchronise the queue */
			err = clWaitForEvents(1, &ev);
			_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

			err = clReleaseEvent(ev);
			_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
		}
		else
		{
			clFlush(peer_transfer_queues[worker->devid]);
			*event = ev;
		}

		if (ret)
		{
			*ret = (event == NULL) ? 0 : -EAGAIN;
		}
	}
	return err;
}

cl_int starpu_opencl_copy_async_sync(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, cl_event *event)
{
	enum starpu_node_kind src_kind = starpu_node_get_kind(src_node);
	enum starpu_node_kind dst_kind = starpu_node_get_kind(dst_node);
	cl_int err;
	int ret;

	if (src_kind == STARPU_OPENCL_RAM && dst_kind == STARPU_CPU_RAM)
	{
		err = starpu_opencl_copy_opencl_to_ram((cl_mem) src, src_node,
						       (void*) (dst + dst_offset), dst_node,
						       size, src_offset, event, &ret);
		_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
		return ret;
	}

	if (src_kind == STARPU_CPU_RAM && dst_kind == STARPU_OPENCL_RAM)
	{
		err = starpu_opencl_copy_ram_to_opencl((void*) (src + src_offset), src_node,
						       (cl_mem) dst, dst_node,
						       size, dst_offset, event, &ret);
		_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
		return ret;
	}

	if (src_kind == STARPU_OPENCL_RAM && (dst_kind == STARPU_CPU_RAM || dst_kind == STARPU_OPENCL_RAM))
	{
		err = starpu_opencl_copy_opencl_to_opencl((cl_mem) src, src_node, src_offset,
							  (cl_mem) dst, dst_node, dst_offset,
							  size, event, &ret);
		_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
		return ret;
	}

	STARPU_ABORT();
}

#if 0
cl_int _starpu_opencl_copy_rect_opencl_to_ram(cl_mem buffer, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *ptr, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, const size_t buffer_origin[3], const size_t host_origin[3],
                                              const size_t region[3], size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                              size_t host_row_pitch, size_t host_slice_pitch, cl_event *event)
{
        cl_int err;
        struct _starpu_worker *worker = _starpu_get_local_worker_key();
        cl_bool blocking;
	double start = 0.;

        blocking = (event == NULL) ? CL_TRUE : CL_FALSE;
        if (event)
                starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
        err = clEnqueueReadBufferRect(out_transfer_queues[worker->devid], buffer, blocking, buffer_origin, host_origin, region, buffer_row_pitch,
                                      buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, 0, NULL, event);
	clFlush(out_transfer_queues[worker->devid]);
        if (event)
                starpu_interface_end_driver_copy_async(src_node, dst_node, start);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        return CL_SUCCESS;
}

cl_int _starpu_opencl_copy_rect_ram_to_opencl(void *ptr, unsigned src_node STARPU_ATTRIBUTE_UNUSED, cl_mem buffer, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, const size_t buffer_origin[3], const size_t host_origin[3],
                                              const size_t region[3], size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                              size_t host_row_pitch, size_t host_slice_pitch, cl_event *event)
{
        cl_int err;
        struct _starpu_worker *worker = _starpu_get_local_worker_key();
        cl_bool blocking;
	double start = 0.;

        blocking = (event == NULL) ? CL_TRUE : CL_FALSE;
        if (event)
                starpu_interface_start_driver_copy_async(src_node, dst_node, &start);
        err = clEnqueueWriteBufferRect(in_transfer_queues[worker->devid], buffer, blocking, buffer_origin, host_origin, region, buffer_row_pitch,
                                       buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, 0, NULL, event);
	clFlush(in_transfer_queues[worker->devid]);
        if (event)
                starpu_interface_end_driver_copy_async(src_node, dst_node, start);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

        return CL_SUCCESS;
}
#endif
#endif /* STARPU_USE_OPENCL */

static size_t _starpu_opencl_get_global_mem_size(int devid)
{
	return global_mem[devid];
}

void _starpu_opencl_init(void)
{
	STARPU_PTHREAD_MUTEX_LOCK(&big_lock);
        if (!init_done)
	{
#ifdef STARPU_SIMGRID
		nb_devices = _starpu_simgrid_get_nbhosts("OpenCL");
#else /* STARPU_USE_OPENCL */
                cl_platform_id platform_id[_STARPU_OPENCL_PLATFORM_MAX];
                cl_uint nb_platforms;
                cl_int err;
                int i;
                cl_device_type device_type = CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_ACCELERATOR;

                _STARPU_DEBUG("Initialising OpenCL\n");

                // Get Platforms
		if (starpu_get_env_number("STARPU_OPENCL_ON_CPUS") > 0)
		     device_type |= CL_DEVICE_TYPE_CPU;
		if (starpu_get_env_number("STARPU_OPENCL_ONLY_ON_CPUS") > 0)
		     device_type = CL_DEVICE_TYPE_CPU;
                err = clGetPlatformIDs(_STARPU_OPENCL_PLATFORM_MAX, platform_id, &nb_platforms);
                if (STARPU_UNLIKELY(err != CL_SUCCESS)) nb_platforms=0;
                _STARPU_DEBUG("Platforms detected: %u\n", nb_platforms);
		_STARPU_DEBUG("CPU device type: %s\n", (device_type&CL_DEVICE_TYPE_CPU)?"requested":"not requested");
		_STARPU_DEBUG("GPU device type: %s\n", (device_type&CL_DEVICE_TYPE_GPU)?"requested":"not requested");
		_STARPU_DEBUG("Accelerator device type: %s\n", (device_type&CL_DEVICE_TYPE_ACCELERATOR)?"requested":"not requested");

                // Get devices
                nb_devices = 0;
                {
			unsigned j;
                        for (j=0; j<nb_platforms; j++)
			{
                                cl_uint num;
				int platform_valid = 1;
				char name[1024], vendor[1024];

				err = clGetPlatformInfo(platform_id[j], CL_PLATFORM_NAME, 1024, name, NULL);
				if (err != CL_SUCCESS)
				{
					STARPU_OPENCL_REPORT_ERROR_WITH_MSG("clGetPlatformInfo NAME", err);
					platform_valid = 0;
				}
				else
				{
					err = clGetPlatformInfo(platform_id[j], CL_PLATFORM_VENDOR, 1024, vendor, NULL);
					if (STARPU_UNLIKELY(err != CL_SUCCESS))
					{
						STARPU_OPENCL_REPORT_ERROR_WITH_MSG("clGetPlatformInfo VENDOR", err);
						platform_valid = 0;
					}
				}
				if(strcmp(name, "SOCL Platform") == 0)
				{
					platform_valid = 0;
					_STARPU_DEBUG("Skipping SOCL Platform\n");
				}
#ifdef STARPU_VERBOSE
				if (platform_valid)
					_STARPU_DEBUG("Platform: %s - %s\n", name, vendor);
				else
					_STARPU_DEBUG("Platform invalid: %s - %s\n", name, vendor);
#endif
				if (platform_valid && nb_devices <= STARPU_MAXOPENCLDEVS)
				{
					err = clGetDeviceIDs(platform_id[j], device_type, STARPU_MAXOPENCLDEVS-nb_devices, STARPU_MAXOPENCLDEVS == nb_devices ? NULL : &devices[nb_devices], &num);
					if (err == CL_DEVICE_NOT_FOUND)
					{
						const cl_device_type all_device_types = CL_DEVICE_TYPE_CPU|CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_ACCELERATOR;
						if (device_type != all_device_types)
						{
							_STARPU_DEBUG("  No devices of the requested type(s) subset detected on this platform\n");
						}
						else
						{
							_STARPU_DEBUG("  No devices detected on this platform\n");
						}
					}
					else
					{
						_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
						_STARPU_DEBUG("  %u devices detected\n", num);
						nb_devices += num;
					}
				}
			}
		}

                // Get location of OpenCl kernel source files
                _starpu_opencl_program_dir = starpu_getenv("STARPU_OPENCL_PROGRAM_DIR");

		if (nb_devices > STARPU_MAXOPENCLDEVS)
		{
			_STARPU_DISP("# Warning: %u OpenCL devices available. Only %d enabled. Use configure option --enable-maxopencldev=xxx to update the maximum value of supported OpenCL devices?\n", nb_devices, STARPU_MAXOPENCLDEVS);
			nb_devices = STARPU_MAXOPENCLDEVS;
		}

                // initialise internal structures
                for(i=0 ; i<nb_devices ; i++)
		{
                        contexts[i] = NULL;
                        queues[i] = NULL;
                        in_transfer_queues[i] = NULL;
                        out_transfer_queues[i] = NULL;
                        peer_transfer_queues[i] = NULL;
                        alloc_queues[i] = NULL;
                }
#endif /* STARPU_USE_OPENCL */

                init_done=1;
        }
	STARPU_PTHREAD_MUTEX_UNLOCK(&big_lock);
}

#ifndef STARPU_SIMGRID
static unsigned _starpu_opencl_get_device_name(int dev, char *name, int lname);
#endif
static int _starpu_opencl_start_job(struct _starpu_job *j, struct _starpu_worker *worker, unsigned char pipeline_idx);
static void _starpu_opencl_stop_job(struct _starpu_job *j, struct _starpu_worker *worker);
static void _starpu_opencl_execute_job(struct starpu_task *task, struct _starpu_worker *worker);

int _starpu_opencl_driver_init(struct _starpu_worker *worker)
{
	int devid = worker->devid;

	_starpu_driver_start(worker, _STARPU_FUT_OPENCL_KEY, 0);

	_starpu_opencl_init_context(devid);

	/* one more time to avoid hacks from third party lib :) */
	_starpu_bind_thread_on_cpu(worker->bindid, worker->workerid, NULL);

	_starpu_opencl_limit_gpu_mem_if_needed(devid);
	_starpu_memory_manager_set_global_memory_size(worker->memory_node, _starpu_opencl_get_global_mem_size(devid));

	float size = (float) global_mem[devid] / (1<<30);

#ifdef STARPU_SIMGRID
	const char *devname = "Simgrid";
#else
	/* get the device's name */
	char devname[64];
	_starpu_opencl_get_device_name(devid, devname, 64);
#endif
	snprintf(worker->name, sizeof(worker->name), "OpenCL %d (%s %.1f GiB)", devid, devname, size);
	snprintf(worker->short_name, sizeof(worker->short_name), "OpenCL %d", devid);
	starpu_pthread_setname(worker->short_name);

	worker->pipeline_length = starpu_get_env_number_default("STARPU_OPENCL_PIPELINE", 2);
	if (worker->pipeline_length > STARPU_MAX_PIPELINE)
	{
		_STARPU_DISP("Warning: STARPU_OPENCL_PIPELINE is %u, but STARPU_MAX_PIPELINE is only %u", worker->pipeline_length, STARPU_MAX_PIPELINE);
		worker->pipeline_length = STARPU_MAX_PIPELINE;
	}
#if !defined(STARPU_SIMGRID) && !defined(STARPU_NON_BLOCKING_DRIVERS)
	if (worker->pipeline_length >= 1)
	{
		/* We need non-blocking drivers, to poll for OPENCL task
		 * termination */
		_STARPU_DISP("Warning: reducing STARPU_OPENCL_PIPELINE to 0 because blocking drivers are enabled (and simgrid is not enabled)\n");
		worker->pipeline_length = 0;
	}
#endif

	_STARPU_DEBUG("OpenCL (%s) dev id %d thread is ready to run on CPU %d !\n", devname, devid, worker->bindid);

	_STARPU_TRACE_WORKER_INIT_END(worker->workerid);

	/* tell the main thread that this one is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
	worker->status = STATUS_UNKNOWN;
	worker->worker_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);

	return 0;
}

int _starpu_opencl_driver_run_once(struct _starpu_worker *worker)
{
	int workerid = worker->workerid;
	unsigned memnode = worker->memory_node;

	struct _starpu_job *j;
	struct starpu_task *task;
	int res;

	int idle_tasks, idle_transfers;

#ifdef STARPU_SIMGRID
	starpu_pthread_wait_reset(&worker->wait);
#endif

	idle_tasks = 0;
	idle_transfers = 0;

	/* First test for transfers pending for next task */
	task = worker->task_transferring;
	if (!task)
		idle_transfers++;
	if (task && worker->nb_buffers_transferred == worker->nb_buffers_totransfer)
	{
		STARPU_RMB();
		_STARPU_TRACE_END_PROGRESS(memnode);
		j = _starpu_get_job_associated_to_task(task);

		_starpu_fetch_task_input_tail(task, j, worker);
		_starpu_set_worker_status(worker, STATUS_UNKNOWN);
		/* Reset it */
		worker->task_transferring = NULL;

		if (worker->ntasks > 1 && !(task->cl->opencl_flags[j->nimpl] & STARPU_OPENCL_ASYNC))
		{
			/* We have to execute a non-asynchronous task but we
			 * still have tasks in the pipeline...  Record it to
			 * prevent more tasks from coming, and do it later */
			worker->pipeline_stuck = 1;
			return 0;
		}

		_starpu_opencl_execute_job(task, worker);
		_STARPU_TRACE_START_PROGRESS(memnode);
	}

	/* Then poll for completed jobs */
	if (worker->ntasks && worker->current_tasks[worker->first_task] != worker->task_transferring)
	{
#ifndef STARPU_SIMGRID
		size_t size;
		int err;
#endif

		/* On-going asynchronous task, check for its termination first */

		task = worker->current_tasks[worker->first_task];

#ifdef STARPU_SIMGRID
		if (!task_finished[worker->devid][worker->first_task])
#else /* !STARPU_SIMGRID */
		cl_int status;
		err = clGetEventInfo(task_events[worker->devid][worker->first_task], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, &size);
		_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
		STARPU_ASSERT(size == sizeof(cl_int));

		if (status != CL_COMPLETE)
#endif /* !STARPU_SIMGRID */
		{
		}
		else
		{
			_STARPU_TRACE_END_PROGRESS(memnode);
#ifndef STARPU_SIMGRID
			err = clReleaseEvent(task_events[worker->devid][worker->first_task]);
			_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
			task_events[worker->devid][worker->first_task] = 0;
#endif

			/* Asynchronous task completed! */
			_starpu_opencl_stop_job(_starpu_get_job_associated_to_task(task), worker);
			/* See next task if any */
			if (worker->ntasks && worker->current_tasks[worker->first_task] != worker->task_transferring)
			{
				task = worker->current_tasks[worker->first_task];
				j = _starpu_get_job_associated_to_task(task);
				if (task->cl->opencl_flags[j->nimpl] & STARPU_OPENCL_ASYNC)
				{
					/* An asynchronous task, it was already queued,
					 * it's now running, record its start time.  */
					_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, starpu_profiling_status_get());
				}
				else
				{
					/* A synchronous task, we have finished flushing the pipeline, we can now at last execute it.  */
					_STARPU_TRACE_EVENT("sync_task");
					_starpu_opencl_execute_job(task, worker);
					_STARPU_TRACE_EVENT("end_sync_task");
					worker->pipeline_stuck = 0;
				}
			}
			_STARPU_TRACE_START_PROGRESS(memnode);
		}
	}
	if (!worker->pipeline_length || worker->ntasks < worker->pipeline_length)
		idle_tasks++;

#if defined(STARPU_NON_BLOCKING_DRIVERS) && !defined(STARPU_SIMGRID)
	if (!idle_tasks)
	{
		/* No task ready yet, no better thing to do than waiting */
		__starpu_datawizard_progress(1, !idle_transfers);
		return 0;
	}
#endif

	res = !idle_tasks || !idle_transfers;
	res |= __starpu_datawizard_progress(1, 1);

	task = _starpu_get_worker_task(worker, workerid, memnode);

#ifdef STARPU_SIMGRID
	if (!res && !task)
		starpu_pthread_wait_wait(&worker->wait);
#endif

	if (task == NULL)
		return 0;

	j = _starpu_get_job_associated_to_task(task);

	worker->current_tasks[(worker->first_task  + worker->ntasks)%STARPU_MAX_PIPELINE] = task;
	worker->ntasks++;
	if (worker->pipeline_length == 0)
	/* _starpu_get_worker_task checks .current_task field if pipeline_length == 0
	 *
	 * TODO: update driver to not use current_tasks[] when pipeline_length == 0,
	 * as for cuda driver */
		worker->current_task = task;

	/* can OpenCL do that task ? */
	if (!_STARPU_OPENCL_MAY_PERFORM(j))
	{
		/* this is not a OpenCL task */
		_starpu_worker_refuse_task(worker, task);
		return 0;
	}

	_STARPU_TRACE_END_PROGRESS(memnode);

	/* Fetch data asynchronously */
	res = _starpu_fetch_task_input(task, j, 1);
	STARPU_ASSERT(res == 0);
	_STARPU_TRACE_START_PROGRESS(memnode);

	return 0;
}

int _starpu_opencl_driver_deinit(struct _starpu_worker *worker)
{
	_STARPU_TRACE_WORKER_DEINIT_START;

	unsigned memnode = worker->memory_node;

	_starpu_handle_all_pending_node_data_requests(memnode);

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	_starpu_malloc_shutdown(memnode);

	unsigned devid   = worker->devid;
        _starpu_opencl_deinit_context(devid);

	worker->worker_is_initialized = 0;
	_STARPU_TRACE_WORKER_DEINIT_END(_STARPU_FUT_OPENCL_KEY);

	return 0;
}

void *_starpu_opencl_worker(void *_arg)
{
	struct _starpu_worker* worker = _arg;

	_starpu_opencl_driver_init(worker);
	_STARPU_TRACE_START_PROGRESS(worker->memory_node);
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_opencl_driver_run_once(worker);
	}
	_starpu_opencl_driver_deinit(worker);
	_STARPU_TRACE_END_PROGRESS(worker->memory_node);

	return NULL;
}

#ifdef STARPU_USE_OPENCL
#ifndef STARPU_SIMGRID
static unsigned _starpu_opencl_get_device_name(int dev, char *name, int lname)
{
	int err;

        if (!init_done)
	{
                _starpu_opencl_init();
        }

	// Get device name
	err = clGetDeviceInfo(devices[dev], CL_DEVICE_NAME, lname, name, NULL);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

	_STARPU_DEBUG("Device %d : [%s]\n", dev, name);
	return EXIT_SUCCESS;
}
#endif
#endif

unsigned _starpu_opencl_get_device_count(void)
{
        if (!init_done)
	{
                _starpu_opencl_init();
        }
	return nb_devices;
}

#ifdef STARPU_USE_OPENCL
cl_device_type _starpu_opencl_get_device_type(int devid)
{
	int err;
	cl_device_type type;

	if (!init_done)
		_starpu_opencl_init();

	err = clGetDeviceInfo(devices[devid], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
	_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);

	return type;
}
#endif /* STARPU_USE_OPENCL */

static int _starpu_opencl_start_job(struct _starpu_job *j, struct _starpu_worker *worker, unsigned char pipeline_idx STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	int profiling = starpu_profiling_status_get();

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	_starpu_set_current_task(task);

	if (worker->ntasks == 1)
	{
		/* We are alone in the pipeline, the kernel will start now, record it */
		_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, profiling);
	}

	starpu_opencl_func_t func = _starpu_task_get_opencl_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT_MSG(func, "when STARPU_OPENCL is defined in 'where', opencl_func or opencl_funcs has to be defined");

	if (_starpu_get_disable_kernels() <= 0)
	{
		_STARPU_TRACE_START_EXECUTING();
#ifdef STARPU_SIMGRID
		double length = NAN;
		int async = task->cl->opencl_flags[j->nimpl] & STARPU_OPENCL_ASYNC;
		int simulate = 1;
		if (cl->flags & STARPU_CODELET_SIMGRID_EXECUTE && !async)
		{
			/* Actually execute function */
			simulate = 0;
			func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
#ifdef STARPU_OPENCL_SIMULATOR
#ifndef CL_PROFILING_CLOCK_CYCLE_COUNT
#ifdef CL_PROFILING_COMMAND_SHAVE_CYCLE_COUNT
#define CL_PROFILING_CLOCK_CYCLE_COUNT CL_PROFILING_COMMAND_SHAVE_CYCLE_COUNT
#else
#error The OpenCL simulator must provide CL_PROFILING_CLOCK_CYCLE_COUNT
#endif
#endif
			struct starpu_profiling_task_info *profiling_info = task->profiling_info;
			STARPU_ASSERT_MSG(profiling_info->used_cycles, "Application kernel must call starpu_opencl_collect_stats to collect simulated time");
#if defined(HAVE_SG_HOST_SPEED) || defined(sg_host_speed)
#  if defined(HAVE_SG_HOST_SELF) || defined(sg_host_self)
			length = ((double) profiling_info->used_cycles)/sg_host_speed(sg_host_self());
#  else
			length = ((double) profiling_info->used_cycles)/sg_host_speed(MSG_host_self());
#  endif
#elif defined HAVE_MSG_HOST_GET_SPEED || defined(MSG_host_get_speed)
			length = ((double) profiling_info->used_cycles)/MSG_host_get_speed(MSG_host_self());
#else
			length = ((double) profiling_info->used_cycles)/MSG_get_host_speed(MSG_host_self());
#endif
			/* And give the simulated time to simgrid */
			simulate = 1;
#endif
		}
		else if (cl->flags & STARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT && !async)
			{
				_SIMGRID_TIMER_BEGIN(1);
				func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
				_SIMGRID_TIMER_END;
				simulate=0;
			}

		if (simulate)
			_starpu_simgrid_submit_job(worker->workerid, j, &worker->perf_arch, length,
						   async ? &task_finished[worker->devid][pipeline_idx] : NULL);
#else
		func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);

		cl_command_queue queue;
		starpu_opencl_get_queue(worker->devid, &queue);
#endif
		_STARPU_TRACE_END_EXECUTING();
	}
	return 0;
}

static void _starpu_opencl_stop_job(struct _starpu_job *j, struct _starpu_worker *worker)
{
	int profiling = starpu_profiling_status_get();

	_starpu_set_current_task(NULL);
	if (worker->pipeline_length)
		worker->current_tasks[worker->first_task] = NULL;
	else
		worker->current_task = NULL;
	worker->first_task = (worker->first_task + 1) % STARPU_MAX_PIPELINE;
	worker->ntasks--;

	_starpu_driver_end_job(worker, j, &worker->perf_arch, 0, profiling);

	struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(worker, j);
	STARPU_ASSERT_MSG(sched_ctx != NULL, "there should be a worker %d in the ctx of this job \n", worker->workerid);
	if(!sched_ctx->sched_policy)
		_starpu_driver_update_job_feedback(j, worker, &sched_ctx->perf_arch, profiling);
	else
		_starpu_driver_update_job_feedback(j, worker, &worker->perf_arch, profiling);

	_starpu_push_task_output(j);

	_starpu_handle_job_termination(j);

}

static void _starpu_opencl_execute_job(struct starpu_task *task, struct _starpu_worker *worker)
{
	int res;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	unsigned char pipeline_idx = (worker->first_task + worker->ntasks - 1)%STARPU_MAX_PIPELINE;

	res = _starpu_opencl_start_job(j, worker, pipeline_idx);

	if (res)
	{
		switch (res)
		{
			case -EAGAIN:
				_STARPU_DISP("ouch, OpenCL could not actually run task %p, putting it back...\n", task);
				_starpu_push_task_to_workers(task);
				STARPU_ABORT();
			default:
				STARPU_ABORT();
		}
	}

	if (task->cl->opencl_flags[j->nimpl] & STARPU_OPENCL_ASYNC)
	{
		/* Record event to synchronize with task termination later */
#ifndef STARPU_SIMGRID
		cl_command_queue queue;
		starpu_opencl_get_queue(worker->devid, &queue);
#endif

		if (worker->pipeline_length == 0)
		{
#ifdef STARPU_SIMGRID
			_starpu_simgrid_wait_tasks(worker->workerid);
#else
			starpu_opencl_get_queue(worker->devid, &queue);
			clFinish(queue);
#endif
			_starpu_opencl_stop_job(j, worker);
		}
		else
		{
#ifndef STARPU_SIMGRID
			int err;
			/* the function clEnqueueMarker is deprecated from
			 * OpenCL version 1.2. We would like to use the new
			 * function clEnqueueMarkerWithWaitList. We could do
			 * it by checking its availability through our own
			 * configure macro HAVE_CLENQUEUEMARKERWITHWAITLIST
			 * and the OpenCL macro CL_VERSION_1_2. However these
			 * 2 macros detect the function availability in the
			 * ICD and not in the device implementation.
			 */
			err = clEnqueueMarker(queue, &task_events[worker->devid][pipeline_idx]);
			_STARPU_OPENCL_CHECK_AND_REPORT_ERROR(err);
			clFlush(queue);
#endif
			_STARPU_TRACE_START_EXECUTING();
		}
	}
	else
	/* Synchronous execution */
	{
		_starpu_opencl_stop_job(j, worker);
	}
}

#ifdef STARPU_USE_OPENCL
int _starpu_run_opencl(struct _starpu_worker *workerarg)
{
	_STARPU_DEBUG("Running OpenCL %u from the application\n", workerarg->devid);

	workerarg->set = NULL;
	workerarg->worker_is_initialized = 0;

	/* Let's go ! */
	_starpu_opencl_worker(workerarg);

	return 0;
}

struct _starpu_driver_ops _starpu_driver_opencl_ops =
{
	.init = _starpu_opencl_driver_init,
	.run = _starpu_run_opencl,
	.run_once = _starpu_opencl_driver_run_once,
	.deinit = _starpu_opencl_driver_deinit
};
#endif

#if defined(STARPU_USE_OPENCL)
unsigned _starpu_opencl_test_request_completion(struct _starpu_async_channel *async_channel)
{
	cl_int event_status;
	cl_event opencl_event = (*async_channel).event.opencl_event;
	if (opencl_event == NULL) STARPU_ABORT();
	cl_int err = clGetEventInfo(opencl_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
	if (event_status < 0)
		STARPU_OPENCL_REPORT_ERROR(event_status);
	if (event_status == CL_COMPLETE)
	{
		err = clReleaseEvent(opencl_event);
		if (STARPU_UNLIKELY(err != CL_SUCCESS)) STARPU_OPENCL_REPORT_ERROR(err);
	}
	return (event_status == CL_COMPLETE);
}

void _starpu_opencl_wait_request_completion(struct _starpu_async_channel *async_channel)
{
	cl_int err;
	if ((*async_channel).event.opencl_event == NULL)
		STARPU_ABORT();
	err = clWaitForEvents(1, &((*async_channel).event.opencl_event));
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
	err = clReleaseEvent((*async_channel).event.opencl_event);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
}

int _starpu_opencl_copy_interface_from_opencl_to_opencl(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_OPENCL_RAM && dst_kind == STARPU_OPENCL_RAM);

	int ret = 1;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
	/* STARPU_OPENCL_RAM -> STARPU_OPENCL_RAM */
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == dst_node || starpu_worker_get_local_memory_node() == src_node);
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_opencl_copy_disabled() || !(copy_methods->opencl_to_opencl_async || copy_methods->any_to_any))
	{
		STARPU_ASSERT(copy_methods->opencl_to_opencl || copy_methods->any_to_any);
		/* this is not associated to a request so it's synchronous */
		if (copy_methods->opencl_to_opencl)
			copy_methods->opencl_to_opencl(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_opencl_node_ops;
		if (copy_methods->opencl_to_opencl_async)
			ret = copy_methods->opencl_to_opencl_async(src_interface, src_node, dst_interface, dst_node, &(req->async_channel.event.opencl_event));
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}
	}
	return ret;
}

int _starpu_opencl_copy_interface_from_opencl_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_OPENCL_RAM && dst_kind == STARPU_CPU_RAM);

	int ret = 1;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
	/* OpenCL -> RAM */
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == src_node);
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_opencl_copy_disabled() || !(copy_methods->opencl_to_ram_async || copy_methods->any_to_any))
	{
		STARPU_ASSERT(copy_methods->opencl_to_ram || copy_methods->any_to_any);
		/* this is not associated to a request so it's synchronous */
		if (copy_methods->opencl_to_ram)
			copy_methods->opencl_to_ram(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_opencl_node_ops;
		if (copy_methods->opencl_to_ram_async)
			ret = copy_methods->opencl_to_ram_async(src_interface, src_node, dst_interface, dst_node, &(req->async_channel.event.opencl_event));
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}
	}
	return ret;
}

int _starpu_opencl_copy_interface_from_cpu_to_opencl(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_OPENCL_RAM);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
	/* STARPU_CPU_RAM -> STARPU_OPENCL_RAM */
	STARPU_ASSERT(starpu_worker_get_local_memory_node() == dst_node);
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_opencl_copy_disabled() || !(copy_methods->ram_to_opencl_async || copy_methods->any_to_any))
	{
		STARPU_ASSERT(copy_methods->ram_to_opencl || copy_methods->any_to_any);
		/* this is not associated to a request so it's synchronous */
		if (copy_methods->ram_to_opencl)
			copy_methods->ram_to_opencl(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		req->async_channel.node_ops = &_starpu_driver_opencl_node_ops;
		if (copy_methods->ram_to_opencl_async)
			ret = copy_methods->ram_to_opencl_async(src_interface, src_node, dst_interface, dst_node, &(req->async_channel.event.opencl_event));
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}
	}
	return ret;
}

int _starpu_opencl_copy_data_from_opencl_to_opencl(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_OPENCL_RAM && dst_kind == STARPU_OPENCL_RAM);

	return starpu_opencl_copy_async_sync(src, src_offset, src_node,
					     dst, dst_offset, dst_node,
					     size,
					     &async_channel->event.opencl_event);
}

int _starpu_opencl_copy_data_from_opencl_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_OPENCL_RAM && dst_kind == STARPU_CPU_RAM);

	return starpu_opencl_copy_async_sync(src, src_offset, src_node,
					     dst, dst_offset, dst_node,
					     size,
					     &async_channel->event.opencl_event);
}

int _starpu_opencl_copy_data_from_cpu_to_opencl(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_OPENCL_RAM);

	return starpu_opencl_copy_async_sync(src, src_offset, src_node,
					     dst, dst_offset, dst_node,
					     size,
					     &async_channel->event.opencl_event);
}
#endif

uintptr_t _starpu_opencl_malloc_on_node(unsigned dst_node, size_t size, int flags)
{
	(void)flags;
	uintptr_t addr = 0;
#ifdef STARPU_SIMGRID
	static uintptr_t last[STARPU_MAXNODES];
	/* Sleep for the allocation */
	STARPU_PTHREAD_MUTEX_LOCK(&opencl_alloc_mutex);
	if (_starpu_simgrid_cuda_malloc_cost())
		starpu_sleep(0.000175);
	if (!last[dst_node])
		last[dst_node] = 1<<10;
	addr = last[dst_node];
	last[dst_node]+=size;
	STARPU_ASSERT(last[dst_node] >= addr);
	STARPU_PTHREAD_MUTEX_UNLOCK(&opencl_alloc_mutex);
#else
	int ret;
	cl_mem ptr;

	ret = starpu_opencl_allocate_memory(starpu_memory_node_get_devid(dst_node), &ptr, size, CL_MEM_READ_WRITE);
	if (ret)
	{
		addr = 0;
	}
	else
	{
		addr = (uintptr_t)ptr;
	}
#endif
	return addr;
}

void _starpu_opencl_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	(void)flags;
	(void)size;
	(void)dst_node;
#ifdef STARPU_SIMGRID
	STARPU_PTHREAD_MUTEX_LOCK(&opencl_alloc_mutex);
	/* Sleep for the free */
	if (_starpu_simgrid_cuda_malloc_cost())
		starpu_sleep(0.000750);
	STARPU_PTHREAD_MUTEX_UNLOCK(&opencl_alloc_mutex);
#else
	cl_int err;
	err = clReleaseMemObject((void*)addr);
	if (STARPU_UNLIKELY(err != CL_SUCCESS))
		STARPU_OPENCL_REPORT_ERROR(err);
#endif
}

int _starpu_opencl_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	(void)node;
	(void)handling_node;
	return 0;
}

#ifdef STARPU_SIMGRID
struct _starpu_node_ops _starpu_driver_opencl_node_ops =
{
	.copy_interface_to[STARPU_UNUSED] = NULL,
	.copy_interface_to[STARPU_CPU_RAM] = NULL,
	.copy_interface_to[STARPU_CUDA_RAM] = NULL,
	.copy_interface_to[STARPU_OPENCL_RAM] = NULL,
	.copy_interface_to[STARPU_DISK_RAM] = NULL,
	.copy_interface_to[STARPU_MIC_RAM] = NULL,
	.copy_interface_to[STARPU_MPI_MS_RAM] = NULL,

	.copy_data_to[STARPU_UNUSED] = NULL,
	.copy_data_to[STARPU_CPU_RAM] = NULL,
	.copy_data_to[STARPU_CUDA_RAM] = NULL,
	.copy_data_to[STARPU_OPENCL_RAM] = NULL,
	.copy_data_to[STARPU_DISK_RAM] = NULL,
	.copy_data_to[STARPU_MIC_RAM] = NULL,
	.copy_data_to[STARPU_MPI_MS_RAM] = NULL,

	.wait_request_completion = NULL,
	.test_request_completion = NULL,
	.is_direct_access_supported = _starpu_opencl_is_direct_access_supported,
	.malloc_on_node = _starpu_opencl_malloc_on_node,
	.free_on_node = _starpu_opencl_free_on_node,
	.name = "opencl driver"
};
#else
struct _starpu_node_ops _starpu_driver_opencl_node_ops =
{
	.copy_interface_to[STARPU_UNUSED] = NULL,
	.copy_interface_to[STARPU_CPU_RAM] = _starpu_opencl_copy_interface_from_opencl_to_cpu,
	.copy_interface_to[STARPU_CUDA_RAM] = NULL,
	.copy_interface_to[STARPU_OPENCL_RAM] = _starpu_opencl_copy_interface_from_opencl_to_opencl,
	.copy_interface_to[STARPU_DISK_RAM] = NULL,
	.copy_interface_to[STARPU_MIC_RAM] = NULL,
	.copy_interface_to[STARPU_MPI_MS_RAM] = NULL,

	.copy_data_to[STARPU_UNUSED] = NULL,
	.copy_data_to[STARPU_CPU_RAM] = _starpu_opencl_copy_data_from_opencl_to_cpu,
	.copy_data_to[STARPU_CUDA_RAM] = NULL,
	.copy_data_to[STARPU_OPENCL_RAM] = _starpu_opencl_copy_data_from_opencl_to_opencl,
	.copy_data_to[STARPU_DISK_RAM] = NULL,
	.copy_data_to[STARPU_MIC_RAM] = NULL,
	.copy_data_to[STARPU_MPI_MS_RAM] = NULL,

	/* TODO: copy2D/3D? */

	.wait_request_completion = _starpu_opencl_wait_request_completion,
	.test_request_completion = _starpu_opencl_test_request_completion,
	.is_direct_access_supported = _starpu_opencl_is_direct_access_supported,
	.malloc_on_node = _starpu_opencl_malloc_on_node,
	.free_on_node = _starpu_opencl_free_on_node,
	.name = "opencl driver"
};
#endif
