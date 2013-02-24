/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

#include <starpu.h>
#include <starpu_cuda.h>
#include <starpu_profiling.h>
#include <common/utils.h>
#include <common/config.h>
#include <core/debug.h>
#include <drivers/driver_common/driver_common.h>
#include "driver_cuda.h"
#include <core/sched_policy.h>
#ifdef HAVE_CUDA_GL_INTEROP_H
#include <cuda_gl_interop.h>
#endif
#include <datawizard/memory_manager.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

/* the number of CUDA devices */
static int ncudagpus;

#ifdef STARPU_USE_CUDA
static cudaStream_t streams[STARPU_NMAXWORKERS];
static cudaStream_t out_transfer_streams[STARPU_NMAXWORKERS];
static cudaStream_t in_transfer_streams[STARPU_NMAXWORKERS];
static cudaStream_t peer_transfer_streams[STARPU_NMAXWORKERS];
static struct cudaDeviceProp props[STARPU_MAXCUDADEVS];
#endif

void
_starpu_cuda_discover_devices (struct _starpu_machine_config *config)
{
	/* Discover the number of CUDA devices. Fill the result in CONFIG. */

#ifdef STARPU_SIMGRID
	config->topology.nhwcudagpus = _starpu_simgrid_get_nbhosts("CUDA");
#else
	int cnt;
	cudaError_t cures;

	cures = cudaGetDeviceCount (&cnt);
	if (STARPU_UNLIKELY(cures != cudaSuccess))
		cnt = 0;
	config->topology.nhwcudagpus = cnt;
#endif
}

#ifdef STARPU_USE_CUDA
/* In case we want to cap the amount of memory available on the GPUs by the
 * mean of the STARPU_LIMIT_CUDA_MEM, we decrease the value of
 * props[devid].totalGlobalMem which is the value returned by
 * _starpu_cuda_get_global_mem_size() to indicate how much memory can
 * be allocated on the device
 */
static void _starpu_cuda_limit_gpu_mem_if_needed(unsigned devid)
{
	int limit;
	char name[30];

	limit = starpu_get_env_number("STARPU_LIMIT_CUDA_MEM");
	if (limit == -1)
	{
		sprintf(name, "STARPU_LIMIT_CUDA_%u_MEM", devid);
		limit = starpu_get_env_number(name);
	}
	if (limit == -1)
	{
		return;
	}

	/* Find the size of the memory on the device */
	size_t totalGlobalMem = props[devid].totalGlobalMem;

	/* How much memory to waste ? */
	size_t to_waste = totalGlobalMem - (size_t)limit*1024*1024;

	props[devid].totalGlobalMem -= to_waste;

	_STARPU_DEBUG("CUDA device %u: Wasting %ld MB / Limit %ld MB / Total %ld MB / Remains %ld MB\n",
			devid, (size_t)to_waste/(1024*1024), (size_t)limit, (size_t)totalGlobalMem/(1024*1024),
			(size_t)(totalGlobalMem - to_waste)/(1024*1024));
}

static size_t _starpu_cuda_get_global_mem_size(unsigned devid)
{
	return (size_t)props[devid].totalGlobalMem;
}

cudaStream_t starpu_cuda_get_local_in_transfer_stream(void)
{
	int worker = starpu_worker_get_id();

	return in_transfer_streams[worker];
}

cudaStream_t starpu_cuda_get_local_out_transfer_stream(void)
{
	int worker = starpu_worker_get_id();

	return out_transfer_streams[worker];
}

cudaStream_t starpu_cuda_get_local_peer_transfer_stream(void)
{
	int worker = starpu_worker_get_id();

	return peer_transfer_streams[worker];
}

cudaStream_t starpu_cuda_get_local_stream(void)
{
	int worker = starpu_worker_get_id();

	return streams[worker];
}

const struct cudaDeviceProp *starpu_cuda_get_device_properties(unsigned workerid)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned devid = config->workers[workerid].devid;
	return &props[devid];
}

void starpu_cuda_set_device(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	STARPU_ABORT();
#else
	cudaError_t cures;
	struct starpu_conf *conf = _starpu_get_machine_config()->conf;
#if !defined(HAVE_CUDA_MEMCPY_PEER) && defined(HAVE_CUDA_GL_INTEROP_H)
	unsigned i;
#endif

#ifdef HAVE_CUDA_MEMCPY_PEER
	if (conf->n_cuda_opengl_interoperability)
	{
		fprintf(stderr, "OpenGL interoperability was requested, but StarPU was built with multithread GPU control support, please reconfigure with --disable-cuda-memcpy-peer but that will disable the memcpy-peer optimizations\n");
		STARPU_ABORT();
	}
#elif !defined(HAVE_CUDA_GL_INTEROP_H)
	if (conf->n_cuda_opengl_interoperability)
	{
		fprintf(stderr,"OpenGL interoperability was requested, but cuda_gl_interop.h could not be compiled, please make sure that OpenGL headers were available before ./configure run.");
		STARPU_ABORT();
	}
#else
	for (i = 0; i < conf->n_cuda_opengl_interoperability; i++)
		if (conf->cuda_opengl_interoperability[i] == devid)
		{
			cures = cudaGLSetGLDevice(devid);
			goto done;
		}
#endif

	cures = cudaSetDevice(devid);

#if !defined(HAVE_CUDA_MEMCPY_PEER) && defined(HAVE_CUDA_GL_INTEROP_H)
done:
#endif
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
#endif
}

#ifndef STARPU_SIMGRID
static void init_context(unsigned devid)
{
	cudaError_t cures;
	int workerid;

	/* TODO: cudaSetDeviceFlag(cudaDeviceMapHost) */

	starpu_cuda_set_device(devid);

#ifdef HAVE_CUDA_MEMCPY_PEER
	if (starpu_get_env_number("STARPU_DISABLE_CUDA_GPU_GPU_DIRECT") == 0)
	{
		int nworkers = starpu_worker_get_count();
		for (workerid = 0; workerid < nworkers; workerid++)
		{
			struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
			if (worker->arch == STARPU_CUDA_WORKER && worker->devid != devid)
			{
				int can;
				cures = cudaDeviceCanAccessPeer(&can, devid, worker->devid);
				if (!cures && can)
				{
					cures = cudaDeviceEnablePeerAccess(worker->devid, 0);
					if (!cures)
						_STARPU_DEBUG("Enabled GPU-Direct %d -> %d\n", worker->devid, devid);
				}
			}
		}
	}
#endif

	/* force CUDA to initialize the context for real */
	cures = cudaFree(0);
	if (STARPU_UNLIKELY(cures))
	{
		if (cures == cudaErrorDevicesUnavailable)
		{
			fprintf(stderr,"All CUDA-capable devices are busy or unavailable\n");
			exit(77);
		}
		STARPU_CUDA_REPORT_ERROR(cures);
	}

	cures = cudaGetDeviceProperties(&props[devid], devid);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
#ifdef HAVE_CUDA_MEMCPY_PEER
	if (props[devid].computeMode == cudaComputeModeExclusive)
	{
		fprintf(stderr, "CUDA is in EXCLUSIVE-THREAD mode, but StarPU was built with multithread GPU control support, please either ask your administrator to use EXCLUSIVE-PROCESS mode (which should really be fine), or reconfigure with --disable-cuda-memcpy-peer but that will disable the memcpy-peer optimizations\n");
		STARPU_ABORT();
	}
#endif

	workerid = starpu_worker_get_id();

	cures = cudaStreamCreate(&streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaStreamCreate(&in_transfer_streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaStreamCreate(&out_transfer_streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaStreamCreate(&peer_transfer_streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}

static void deinit_context(int workerid)
{
	cudaError_t cures;

	cudaStreamDestroy(streams[workerid]);
	cudaStreamDestroy(in_transfer_streams[workerid]);
	cudaStreamDestroy(out_transfer_streams[workerid]);
	cudaStreamDestroy(peer_transfer_streams[workerid]);

	/* cleanup the runtime API internal stuffs (which CUBLAS is using) */
	cures = cudaThreadExit();
	if (cures)
		STARPU_CUDA_REPORT_ERROR(cures);
}
#endif /* !SIMGRID */

#endif /* STARPU_USE_CUDA */

/* Return the number of devices usable in the system.
 * The value returned cannot be greater than MAXCUDADEVS */

unsigned _starpu_get_cuda_device_count(void)
{
#ifdef STARPU_SIMGRID
	return _starpu_simgrid_get_nbhosts("CUDA");
#else
	int cnt;

	cudaError_t cures;
	cures = cudaGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures))
		 return 0;

	if (cnt > STARPU_MAXCUDADEVS)
	{
		fprintf(stderr, "# Warning: %d CUDA devices available. Only %d enabled. Use configure option --enable-maxcudadev=xxx to update the maximum value of supported CUDA devices.\n", cnt, STARPU_MAXCUDADEVS);
		cnt = STARPU_MAXCUDADEVS;
	}
	return (unsigned)cnt;
#endif
}

void _starpu_init_cuda(void)
{
	ncudagpus = _starpu_get_cuda_device_count();
	STARPU_ASSERT(ncudagpus <= STARPU_MAXCUDADEVS);
}

static int execute_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *args)
{
	int ret;
	uint32_t mask = 0;

	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	struct timespec codelet_start, codelet_end;

	int profiling = starpu_profiling_status_get();

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	ret = _starpu_fetch_task_input(j, mask);
	if (ret != 0)
	{
		/* there was not enough memory, so the input of
		 * the codelet cannot be fetched ... put the
		 * codelet back, and try it later */
		return -EAGAIN;
	}

	_starpu_driver_start_job(args, j, &codelet_start, 0, profiling);

#if defined(HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
	/* We make sure we do manipulate the proper device */
	starpu_cuda_set_device(args->devid);
#endif

	starpu_cuda_func_t func = _starpu_task_get_cuda_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT(func);

#ifdef STARPU_SIMGRID
	_starpu_simgrid_execute_job(j, args->perf_arch, NAN);
#else
	func(task->interfaces, task->cl_arg);
#endif

	_starpu_driver_end_job(args, j, args->perf_arch, &codelet_end, 0, profiling);

	_starpu_driver_update_job_feedback(j, args, args->perf_arch, &codelet_start, &codelet_end, profiling);

	_starpu_push_task_output(j, mask);

	return 0;
}

static struct _starpu_worker*
_starpu_get_worker_from_driver(struct starpu_driver *d)
{
	unsigned nworkers = starpu_worker_get_count();
	unsigned  workerid;
	for (workerid = 0; workerid < nworkers; workerid++)
	{
		if (starpu_worker_get_type(workerid) == d->type)
		{
			struct _starpu_worker *worker;
			worker = _starpu_get_worker_struct(workerid);
			if (worker->devid == d->id.cuda_id)
				return worker;
		}
	}

	return NULL;
}

/* XXX Should this be merged with _starpu_init_cuda ? */
int _starpu_cuda_driver_init(struct starpu_driver *d)
{
	struct _starpu_worker* args = _starpu_get_worker_from_driver(d);
	STARPU_ASSERT(args);
	unsigned devid = args->devid;

	_starpu_worker_init(args, _STARPU_FUT_CUDA_KEY);

#ifndef STARPU_SIMGRID
	init_context(devid);
#endif

#ifdef STARPU_USE_CUDA
	_starpu_cuda_limit_gpu_mem_if_needed(devid);
#endif
	_starpu_memory_manager_set_global_memory_size(args->memory_node, _starpu_cuda_get_global_mem_size(devid));

	/* one more time to avoid hacks from third party lib :) */
	_starpu_bind_thread_on_cpu(args->config, args->bindid);

	args->status = STATUS_UNKNOWN;

#ifdef STARPU_SIMGRID
	const char *devname = "Simgrid";
	snprintf(args->name, sizeof(args->name), "CUDA %u (%s TODO GiB)", devid, devname);
#else
	/* get the device's name */
	char devname[128];
	strncpy(devname, props[devid].name, 128);
	float size = (float) props[devid].totalGlobalMem / (1<<30);

#ifdef STARPU_HAVE_BUSID
#ifdef STARPU_HAVE_DOMAINID
	if (props[devid].pciDomainID)
		snprintf(args->name, sizeof(args->name), "CUDA %u (%s %.1f GiB %04x:%02x:%02x.0)", devid, devname, size, props[devid].pciDomainID, props[devid].pciBusID, props[devid].pciDeviceID);
	else
#endif
		snprintf(args->name, sizeof(args->name), "CUDA %u (%s %.1f GiB %02x:%02x.0)", devid, devname, size, props[devid].pciBusID, props[devid].pciDeviceID);
#else
	snprintf(args->name, sizeof(args->name), "CUDA %u (%s %.1f GiB)", devid, devname, size);
#endif
#endif
	snprintf(args->short_name, sizeof(args->short_name), "CUDA %u", devid);
	_STARPU_DEBUG("cuda (%s) dev id %u thread is ready to run on CPU %d !\n", devname, devid, args->bindid);

	_STARPU_TRACE_WORKER_INIT_END

	/* tell the main thread that this one is ready */
	_STARPU_PTHREAD_MUTEX_LOCK(&args->mutex);
	args->worker_is_initialized = 1;
	_STARPU_PTHREAD_COND_SIGNAL(&args->ready_cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&args->mutex);

	return 0;
}

int _starpu_cuda_driver_run_once(struct starpu_driver *d)
{
	struct _starpu_worker* args = _starpu_get_worker_from_driver(d);
	STARPU_ASSERT(args);

	unsigned memnode = args->memory_node;
	int workerid = args->workerid;

	_STARPU_TRACE_START_PROGRESS(memnode);
	_starpu_datawizard_progress(memnode, 1);
	_STARPU_TRACE_END_PROGRESS(memnode);

	struct starpu_task *task;
	struct _starpu_job *j = NULL;

	task = _starpu_get_worker_task(args, workerid, memnode);

	if (!task)
		return 0;

	j = _starpu_get_job_associated_to_task(task);

	/* can CUDA do that task ? */
	if (!_STARPU_CUDA_MAY_PERFORM(j))
	{
		/* this is neither a cuda or a cublas task */
		_starpu_push_task_to_workers(task);
		return 0;
	}

	_starpu_set_current_task(task);
	args->current_task = j->task;

	int res = execute_job_on_cuda(j, args);

	_starpu_set_current_task(NULL);
	args->current_task = NULL;

	if (res)
	{
		switch (res)
		{
			case -EAGAIN:
				_STARPU_DISP("ouch, put the codelet %p back ... \n", j);
				_starpu_push_task_to_workers(task);
				STARPU_ABORT();
			default:
				STARPU_ABORT();
		}
	}

	_starpu_handle_job_termination(j);

	return 0;
}

int _starpu_cuda_driver_deinit(struct starpu_driver *d)
{
	struct _starpu_worker* args = _starpu_get_worker_from_driver(d);
	STARPU_ASSERT(args);
	unsigned memnode = args->memory_node;

	_STARPU_TRACE_WORKER_DEINIT_START

	_starpu_handle_all_pending_node_data_requests(memnode);

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

#ifndef STARPU_SIMGRID
	deinit_context(args->workerid);
#endif

	_STARPU_TRACE_WORKER_DEINIT_END(_STARPU_FUT_CUDA_KEY);

	return 0;
}

void *_starpu_cuda_worker(void *arg)
{
	struct _starpu_worker* args = arg;
	struct starpu_driver d =
		{
			.type       = STARPU_CUDA_WORKER,
			.id.cuda_id = args->devid
		};

	_starpu_cuda_driver_init(&d);
	while (_starpu_machine_is_running())
		_starpu_cuda_driver_run_once(&d);
	_starpu_cuda_driver_deinit(&d);

	return NULL;
}

#ifdef STARPU_USE_CUDA
void starpu_cublas_report_error(const char *func, const char *file, int line, cublasStatus status)
{
	char *errormsg;
	switch (status)
	{
		case CUBLAS_STATUS_SUCCESS:
			errormsg = "success";
			break;
		case CUBLAS_STATUS_NOT_INITIALIZED:
			errormsg = "not initialized";
			break;
		case CUBLAS_STATUS_ALLOC_FAILED:
			errormsg = "alloc failed";
			break;
		case CUBLAS_STATUS_INVALID_VALUE:
			errormsg = "invalid value";
			break;
		case CUBLAS_STATUS_ARCH_MISMATCH:
			errormsg = "arch mismatch";
			break;
		case CUBLAS_STATUS_EXECUTION_FAILED:
			errormsg = "execution failed";
			break;
		case CUBLAS_STATUS_INTERNAL_ERROR:
			errormsg = "internal error";
			break;
		default:
			errormsg = "unknown error";
			break;
	}
	fprintf(stderr, "oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ABORT();
}

void starpu_cuda_report_error(const char *func, const char *file, int line, cudaError_t status)
{
	const char *errormsg = cudaGetErrorString(status);
	printf("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ABORT();
}

int
starpu_cuda_copy_async_sync(void *src_ptr, unsigned src_node,
			    void *dst_ptr, unsigned dst_node,
			    size_t ssize, cudaStream_t stream,
			    enum cudaMemcpyKind kind)
{
#ifdef HAVE_CUDA_MEMCPY_PEER
	int peer_copy = 0;
	int src_dev = -1, dst_dev = -1;
#endif
	cudaError_t cures = 0;

	if (kind == cudaMemcpyDeviceToDevice && src_node != dst_node)
	{
#ifdef HAVE_CUDA_MEMCPY_PEER
		peer_copy = 1;
		src_dev = _starpu_memory_node_get_devid(src_node);
		dst_dev = _starpu_memory_node_get_devid(dst_node);
#else
		STARPU_ABORT();
#endif
	}

	if (stream)
	{
		_STARPU_TRACE_START_DRIVER_COPY_ASYNC(src_node, dst_node);
#ifdef HAVE_CUDA_MEMCPY_PEER
		if (peer_copy)
		{
			cures = cudaMemcpyPeerAsync((char *) dst_ptr, dst_dev,
						    (char *) src_ptr, src_dev,
						    ssize, stream);
		}
		else
#endif
		{
			cures = cudaMemcpyAsync((char *)dst_ptr, (char *)src_ptr, ssize, kind, stream);
		}
		_STARPU_TRACE_END_DRIVER_COPY_ASYNC(src_node, dst_node);
	}

	/* Test if the asynchronous copy has failed or if the caller only asked for a synchronous copy */
	if (stream == NULL || cures)
	{
		/* do it in a synchronous fashion */
#ifdef HAVE_CUDA_MEMCPY_PEER
		if (peer_copy)
		{
			cures = cudaMemcpyPeer((char *) dst_ptr, dst_dev,
					       (char *) src_ptr, src_dev,
					       ssize);
		}
		else
#endif
		{
			cures = cudaMemcpy((char *)dst_ptr, (char *)src_ptr, ssize, kind);
		}


		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		return 0;
	}

	return -EAGAIN;
}

int _starpu_run_cuda(struct starpu_driver *d)
{
	STARPU_ASSERT(d && d->type == STARPU_CUDA_WORKER);

	int workerid = starpu_worker_get_by_devid(STARPU_CUDA_WORKER, d->id.cuda_id);

	_STARPU_DEBUG("Running cuda %u from the application\n", d->id.cuda_id);

	struct _starpu_worker *workerarg = _starpu_get_worker_struct(workerid);

	workerarg->set = NULL;
	workerarg->worker_is_initialized = 0;

	/* Let's go ! */
	_starpu_cuda_worker(workerarg);

	/* XXX: Should we wait for the driver to be ready, as it is done when
	 * launching it the usual way ? Cf. the end of _starpu_launch_drivers()
	 */

	return 0;
}

#endif /* STARPU_USE_CUDA */
