/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

#ifdef STARPU_USE_CUDA
#if CUDART_VERSION >= 5000
/* Avoid letting our streams spuriously synchonize with the NULL stream */
#define starpu_cudaStreamCreate(stream) cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking)
#else
#define starpu_cudaStreamCreate(stream) cudaStreamCreate(stream)
#endif
#endif

/* the number of CUDA devices */
static int ncudagpus = -1;

static size_t global_mem[STARPU_MAXCUDADEVS];
#ifdef STARPU_USE_CUDA
static cudaStream_t streams[STARPU_NMAXWORKERS];
static char used_stream[STARPU_NMAXWORKERS];
static cudaStream_t out_transfer_streams[STARPU_MAXCUDADEVS];
static cudaStream_t in_transfer_streams[STARPU_MAXCUDADEVS];
/* Note: streams are not thread-safe, so we define them for each CUDA worker
 * emitting a GPU-GPU transfer */
static cudaStream_t in_peer_transfer_streams[STARPU_MAXCUDADEVS][STARPU_MAXCUDADEVS];
static struct cudaDeviceProp props[STARPU_MAXCUDADEVS];
#ifndef STARPU_SIMGRID
static cudaEvent_t task_events[STARPU_NMAXWORKERS][STARPU_MAX_PIPELINE];
#endif
#endif /* STARPU_USE_CUDA */
#ifdef STARPU_SIMGRID
static unsigned task_finished[STARPU_NMAXWORKERS][STARPU_MAX_PIPELINE];
static starpu_pthread_mutex_t task_mutex[STARPU_NMAXWORKERS][STARPU_MAX_PIPELINE];
static starpu_pthread_cond_t task_cond[STARPU_NMAXWORKERS][STARPU_MAX_PIPELINE];
#endif /* STARPU_SIMGRID */

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

/* In case we want to cap the amount of memory available on the GPUs by the
 * mean of the STARPU_LIMIT_CUDA_MEM, we decrease the value of
 * global_mem[devid] which is the value returned by
 * _starpu_cuda_get_global_mem_size() to indicate how much memory can
 * be allocated on the device
 */
static void _starpu_cuda_limit_gpu_mem_if_needed(unsigned devid)
{
	starpu_ssize_t limit;
	size_t STARPU_ATTRIBUTE_UNUSED totalGlobalMem = 0;
	size_t STARPU_ATTRIBUTE_UNUSED to_waste = 0;

#ifdef STARPU_SIMGRID
	totalGlobalMem = _starpu_simgrid_get_memsize("CUDA", devid);
#elif defined(STARPU_USE_CUDA)
	/* Find the size of the memory on the device */
	totalGlobalMem = props[devid].totalGlobalMem;
#endif

	limit = starpu_get_env_number("STARPU_LIMIT_CUDA_MEM");
	if (limit == -1)
	{
		char name[30];
		sprintf(name, "STARPU_LIMIT_CUDA_%u_MEM", devid);
		limit = starpu_get_env_number(name);
	}
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (limit == -1)
	{
		/* Use 90% of the available memory by default.  */
		limit = totalGlobalMem / (1024*1024) * 0.9;
	}
#endif

	global_mem[devid] = limit * 1024*1024;

#ifdef STARPU_USE_CUDA
	/* How much memory to waste ? */
	to_waste = totalGlobalMem - global_mem[devid];

	props[devid].totalGlobalMem -= to_waste;
#endif /* STARPU_USE_CUDA */

	_STARPU_DEBUG("CUDA device %u: Wasting %ld MB / Limit %ld MB / Total %ld MB / Remains %ld MB\n",
			devid, (long) to_waste/(1024*1024), (long) limit, (long) totalGlobalMem/(1024*1024),
			(long) (totalGlobalMem - to_waste)/(1024*1024));
}

#ifdef STARPU_USE_CUDA
cudaStream_t starpu_cuda_get_local_in_transfer_stream()
{
	int worker = starpu_worker_get_id();
	int devid = starpu_worker_get_devid(worker);
	cudaStream_t stream;

	stream = in_transfer_streams[devid];
	STARPU_ASSERT(stream);
	return stream;
}

cudaStream_t starpu_cuda_get_local_out_transfer_stream()
{
	int worker = starpu_worker_get_id();
	int devid = starpu_worker_get_devid(worker);
	cudaStream_t stream;

	stream = out_transfer_streams[devid];
	STARPU_ASSERT(stream);
	return stream;
}

cudaStream_t starpu_cuda_get_peer_transfer_stream(unsigned src_node, unsigned dst_node)
{
	int src_devid = _starpu_memory_node_get_devid(src_node);
	int dst_devid = _starpu_memory_node_get_devid(dst_node);
	cudaStream_t stream;

	stream = in_peer_transfer_streams[src_devid][dst_devid];
	STARPU_ASSERT(stream);
	return stream;
}

cudaStream_t starpu_cuda_get_local_stream(void)
{
	int worker = starpu_worker_get_id();

	used_stream[worker] = 1;
	return streams[worker];
}

const struct cudaDeviceProp *starpu_cuda_get_device_properties(unsigned workerid)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	unsigned devid = config->workers[workerid].devid;
	return &props[devid];
}
#endif /* STARPU_USE_CUDA */

void starpu_cuda_set_device(unsigned devid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	STARPU_ABORT();
#else
	cudaError_t cures;
	struct starpu_conf *conf = &_starpu_get_machine_config()->conf;
#if !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && defined(HAVE_CUDA_GL_INTEROP_H)
	unsigned i;
#endif

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	if (conf->n_cuda_opengl_interoperability)
	{
		_STARPU_MSG("OpenGL interoperability was requested, but StarPU was built with multithread GPU control support, please reconfigure with --disable-cuda-memcpy-peer but that will disable the memcpy-peer optimizations\n");
		STARPU_ABORT();
	}
#elif !defined(HAVE_CUDA_GL_INTEROP_H)
	if (conf->n_cuda_opengl_interoperability)
	{
		_STARPU_MSG("OpenGL interoperability was requested, but cuda_gl_interop.h could not be compiled, please make sure that OpenGL headers were available before ./configure run.");
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

#if !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && defined(HAVE_CUDA_GL_INTEROP_H)
done:
#endif
	if (STARPU_UNLIKELY(cures
#ifdef STARPU_OPENMP
		/* When StarPU is used as Open Runtime support,
		 * starpu_omp_shutdown() will usually be called from a
		 * destructor, in which case cudaThreadExit() reports a
		 * cudaErrorCudartUnloading here. There should not
		 * be any remaining tasks running at this point so
		 * we can probably ignore it without much consequences. */
		&& cures != cudaErrorCudartUnloading
#endif /* STARPU_OPENMP */
				))
		STARPU_CUDA_REPORT_ERROR(cures);
#endif
}

#ifndef STARPU_SIMGRID
static void init_device_context(unsigned devid)
{
	cudaError_t cures;

	/* TODO: cudaSetDeviceFlag(cudaDeviceMapHost) */

	starpu_cuda_set_device(devid);

#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	if (starpu_get_env_number("STARPU_ENABLE_CUDA_GPU_GPU_DIRECT") != 0)
	{
		int nworkers = starpu_worker_get_count();
		int workerid;
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
			_STARPU_MSG("All CUDA-capable devices are busy or unavailable\n");
			exit(77);
		}
		STARPU_CUDA_REPORT_ERROR(cures);
	}

	cures = cudaGetDeviceProperties(&props[devid], devid);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	if (props[devid].computeMode == cudaComputeModeExclusive)
	{
		_STARPU_MSG("CUDA is in EXCLUSIVE-THREAD mode, but StarPU was built with multithread GPU control support, please either ask your administrator to use EXCLUSIVE-PROCESS mode (which should really be fine), or reconfigure with --disable-cuda-memcpy-peer but that will disable the memcpy-peer optimizations\n");
		STARPU_ABORT();
	}
#endif

	cures = starpu_cudaStreamCreate(&in_transfer_streams[devid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = starpu_cudaStreamCreate(&out_transfer_streams[devid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	unsigned i;
	for (i = 0; i < ncudagpus; i++)
	{
		cures = starpu_cudaStreamCreate(&in_peer_transfer_streams[i][devid]);
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}
}
#endif /* !STARPU_SIMGRID */

static void init_worker_context(unsigned workerid)
{
	int j;
#ifdef STARPU_SIMGRID
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
	{
		task_finished[workerid][j] = 0;
		STARPU_PTHREAD_MUTEX_INIT(&task_mutex[workerid][j], NULL);
		STARPU_PTHREAD_COND_INIT(&task_cond[workerid][j], NULL);
	}
#else /* !STARPU_SIMGRID */
	cudaError_t cures;

	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
	{
		cures = cudaEventCreateWithFlags(&task_events[workerid][j], cudaEventDisableTiming);
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}

	cures = starpu_cudaStreamCreate(&streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

#endif /* !STARPU_SIMGRID */
}

#ifndef STARPU_SIMGRID
static void deinit_device_context(unsigned devid)
{
	unsigned i;

	cudaStreamDestroy(in_transfer_streams[devid]);
	cudaStreamDestroy(out_transfer_streams[devid]);

	for (i = 0; i < ncudagpus; i++)
	{
		cudaStreamDestroy(in_peer_transfer_streams[i][devid]);
	}
}
#endif /* !STARPU_SIMGRID */

static void deinit_worker_context(unsigned workerid)
{
	unsigned j;
#ifdef STARPU_SIMGRID
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
	{
		STARPU_PTHREAD_MUTEX_DESTROY(&task_mutex[workerid][j]);
		STARPU_PTHREAD_COND_DESTROY(&task_cond[workerid][j]);
	}
#else /* STARPU_SIMGRID */
	for (j = 0; j < STARPU_MAX_PIPELINE; j++)
		cudaEventDestroy(task_events[workerid][j]);
	cudaStreamDestroy(streams[workerid]);
#endif /* STARPU_SIMGRID */
}

static size_t _starpu_cuda_get_global_mem_size(unsigned devid)
{
	return global_mem[devid];
}


/* Return the number of devices usable in the system.
 * The value returned cannot be greater than MAXCUDADEVS */

unsigned _starpu_get_cuda_device_count(void)
{
	int cnt;
#ifdef STARPU_SIMGRID
	cnt = _starpu_simgrid_get_nbhosts("CUDA");
#else
	cudaError_t cures;
	cures = cudaGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures))
		 return 0;
#endif

	if (cnt > STARPU_MAXCUDADEVS)
	{
		_STARPU_MSG("# Warning: %d CUDA devices available. Only %d enabled. Use configure option --enable-maxcudadev=xxx to update the maximum value of supported CUDA devices.\n", cnt, STARPU_MAXCUDADEVS);
		cnt = STARPU_MAXCUDADEVS;
	}
	return (unsigned)cnt;
}

void _starpu_init_cuda(void)
{
	if (ncudagpus < 0)
	{
		ncudagpus = _starpu_get_cuda_device_count();
		STARPU_ASSERT(ncudagpus <= STARPU_MAXCUDADEVS);
	}
}

static int start_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker, unsigned char pipeline_idx STARPU_ATTRIBUTE_UNUSED)
{
	int ret;

	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	int profiling = starpu_profiling_status_get();

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	_starpu_set_current_task(task);

	ret = _starpu_fetch_task_input(j);
	if (ret != 0)
	{
		/* there was not enough memory, so the input of
		 * the codelet cannot be fetched ... put the
		 * codelet back, and try it later */
		return -EAGAIN;
	}

	if (worker->ntasks == 1)
	{
		/* We are alone in the pipeline, the kernel will start now, record it */
		_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, profiling);
	}

#if defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
	/* We make sure we do manipulate the proper device */
	starpu_cuda_set_device(worker->devid);
#endif

	starpu_cuda_func_t func = _starpu_task_get_cuda_nth_implementation(cl, j->nimpl);
	STARPU_ASSERT_MSG(func, "when STARPU_CUDA is defined in 'where', cuda_func or cuda_funcs has to be defined");

	if (_starpu_get_disable_kernels() <= 0)
	{
		_STARPU_TRACE_START_EXECUTING();
#ifdef STARPU_SIMGRID
		int async = task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC;
		unsigned workerid = worker->workerid;
		if (cl->flags & STARPU_CODELET_SIMGRID_EXECUTE && !async)
			func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
		else
			_starpu_simgrid_submit_job(workerid, j, &worker->perf_arch, NAN,
				async ? &task_finished[workerid][pipeline_idx] : NULL,
				async ? &task_mutex[workerid][pipeline_idx] : NULL,
				async ? &task_cond[workerid][pipeline_idx] : NULL);
#else
		func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
#endif
		_STARPU_TRACE_END_EXECUTING();
	}

	return 0;
}

static void finish_job_on_cuda(struct _starpu_job *j, struct _starpu_worker *worker)
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
	if(!sched_ctx)
		sched_ctx = _starpu_get_sched_ctx_struct(j->task->sched_ctx);

	if(!sched_ctx->sched_policy)
		_starpu_driver_update_job_feedback(j, worker, &sched_ctx->perf_arch, profiling);
	else
		_starpu_driver_update_job_feedback(j, worker, &worker->perf_arch, profiling);

	_starpu_push_task_output(j);

	_starpu_handle_job_termination(j);
}

/* Execute a job, up to completion for synchronous jobs */
static void execute_job_on_cuda(struct starpu_task *task, struct _starpu_worker *worker)
{
	int workerid = worker->workerid;
	int res;

	struct _starpu_job *j = _starpu_get_job_associated_to_task(task);

	unsigned char pipeline_idx = (worker->first_task + worker->ntasks - 1)%STARPU_MAX_PIPELINE;

	res = start_job_on_cuda(j, worker, pipeline_idx);

	if (res)
	{
		switch (res)
		{
			case -EAGAIN:
				_STARPU_DISP("ouch, CUDA could not actually run task %p, putting it back...\n", task);
				_starpu_push_task_to_workers(task);
				STARPU_ABORT();
			default:
				STARPU_ABORT();
		}
	}

#ifndef STARPU_SIMGRID
	if (!used_stream[workerid])
	{
		used_stream[workerid] = 1;
		_STARPU_DISP("Warning: starpu_cuda_get_local_stream() was not used to submit kernel to CUDA on worker %d. CUDA will thus introduce a lot of useless synchronizations, which will prevent proper overlapping of data transfers and kernel execution. See the CUDA-specific part of the 'Check List When Performance Are Not There' of the StarPU handbook\n", workerid);
	}
#endif

	if (task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC)
	{
		if (worker->pipeline_length == 0)
		{
#ifdef STARPU_SIMGRID
			_starpu_simgrid_wait_tasks(workerid);
#else
			/* Forced synchronous execution */
			cudaStreamSynchronize(starpu_cuda_get_local_stream());
#endif
			finish_job_on_cuda(j, worker);
		}
		else
		{
#ifndef STARPU_SIMGRID
			/* Record event to synchronize with task termination later */
			cudaError_t cures = cudaEventRecord(task_events[workerid][pipeline_idx], starpu_cuda_get_local_stream());
			if (STARPU_UNLIKELY(cures))
				STARPU_CUDA_REPORT_ERROR(cures);
#endif
#ifdef STARPU_USE_FXT
			int k;
			for (k = 0; k < (int) worker->set->nworkers; k++)
				if (worker->set->workers[k].ntasks == worker->set->workers[k].pipeline_length)
					break;
			if (k == (int) worker->set->nworkers)
				/* Everybody busy */
				_STARPU_TRACE_START_EXECUTING();
#endif
		}
	}
	else
	/* Synchronous execution */
	{
#if !defined(STARPU_SIMGRID)
		STARPU_ASSERT_MSG(cudaStreamQuery(starpu_cuda_get_local_stream()) == cudaSuccess, "Unless when using the STARPU_CUDA_ASYNC flag, CUDA codelets have to wait for termination of their kernels on the starpu_cuda_get_local_stream() stream");
#endif
		finish_job_on_cuda(j, worker);
	}
}

/* XXX Should this be merged with _starpu_init_cuda ? */
int _starpu_cuda_driver_init(struct _starpu_worker_set *worker_set)
{
	struct _starpu_worker *worker0 = &worker_set->workers[0];
	int lastdevid = -1;
	unsigned i;

	_starpu_driver_start(worker0, _STARPU_FUT_CUDA_KEY, 0);
	_starpu_set_local_worker_set_key(worker_set);

#ifdef STARPU_USE_FXT
	for (i = 1; i < worker_set->nworkers; i++)
		_starpu_worker_start(&worker_set->workers[i], _STARPU_FUT_CUDA_KEY, 0);
#endif

	for (i = 0; i < worker_set->nworkers; i++)
	{
		struct _starpu_worker *worker = &worker_set->workers[i];
		unsigned devid = worker->devid;
		unsigned memnode = worker->memory_node;
		if ((int) devid == lastdevid)
			/* Already initialized */
			continue;
		lastdevid = devid;
#ifndef STARPU_SIMGRID
		init_device_context(devid);
#endif

#ifdef STARPU_SIMGRID
		STARPU_ASSERT_MSG(worker_set->nworkers == 1, "Simgrid mode does not support concurrent kernel execution yet\n");
#else /* !STARPU_SIMGRID */
		if (worker_set->nworkers > 1 && props[devid].concurrentKernels == 0)
			_STARPU_DISP("Warning: STARPU_NWORKER_PER_CUDA is %u, but the device does not support concurrent kernel execution!\n", worker_set->nworkers);
#endif /* !STARPU_SIMGRID */

		_starpu_cuda_limit_gpu_mem_if_needed(devid);
		_starpu_memory_manager_set_global_memory_size(memnode, _starpu_cuda_get_global_mem_size(devid));
	}

	/* one more time to avoid hacks from third party lib :) */
	_starpu_bind_thread_on_cpu(worker0->config, worker0->bindid, worker0->workerid);

	for (i = 0; i < worker_set->nworkers; i++)
	{
		struct _starpu_worker *worker = &worker_set->workers[i];
		unsigned devid = worker->devid;
		unsigned workerid = worker->workerid;

		float size = (float) global_mem[devid] / (1<<30);
#ifdef STARPU_SIMGRID
		const char *devname = "Simgrid";
#else
		/* get the device's name */
		char devname[128];
		strncpy(devname, props[devid].name, 127);
		devname[127] = 0;
#endif

#if defined(STARPU_HAVE_BUSID) && !defined(STARPU_SIMGRID)
#if defined(STARPU_HAVE_DOMAINID) && !defined(STARPU_SIMGRID)
		if (props[devid].pciDomainID)
			snprintf(worker->name, sizeof(worker->name), "CUDA %u.%u (%s %.1f GiB %04x:%02x:%02x.0)", devid, i, devname, size, props[devid].pciDomainID, props[devid].pciBusID, props[devid].pciDeviceID);
		else
#endif
			snprintf(worker->name, sizeof(worker->name), "CUDA %u.%u (%s %.1f GiB %02x:%02x.0)", devid, i, devname, size, props[devid].pciBusID, props[devid].pciDeviceID);
#else
		snprintf(worker->name, sizeof(worker->name), "CUDA %u.%u (%s %.1f GiB)", devid, i, devname, size);
#endif
		snprintf(worker->short_name, sizeof(worker->short_name), "CUDA %u.%u", devid, i);
		_STARPU_DEBUG("cuda (%s) dev id %u worker %u thread is ready to run on CPU %d !\n", devname, devid, i, worker->bindid);

		worker->pipeline_length = starpu_get_env_number_default("STARPU_CUDA_PIPELINE", 2);
		if (worker->pipeline_length > STARPU_MAX_PIPELINE)
		{
			_STARPU_DISP("Warning: STARPU_CUDA_PIPELINE is %u, but STARPU_MAX_PIPELINE is only %u", worker->pipeline_length, STARPU_MAX_PIPELINE);
			worker->pipeline_length = STARPU_MAX_PIPELINE;
		}
#if defined(STARPU_SIMGRID) && defined(STARPU_NON_BLOCKING_DRIVERS)
		if (worker->pipeline_length >= 1)
		{
			/* We need blocking drivers, otherwise idle drivers
			 * would keep consuming real CPU time while just
			 * polling for task termination */
			_STARPU_DISP("Warning: reducing STARPU_CUDA_PIPELINE to 0 because simgrid is enabled and blocking drivers are not enabled\n");
			worker->pipeline_length = 0;
		}
#endif
#if !defined(STARPU_SIMGRID) && !defined(STARPU_NON_BLOCKING_DRIVERS)
		if (worker->pipeline_length >= 1)
		{
			/* We need non-blocking drivers, to poll for CUDA task
			 * termination */
			_STARPU_DISP("Warning: reducing STARPU_CUDA_PIPELINE to 0 because blocking drivers are enabled (and simgrid is not enabled)\n");
			worker->pipeline_length = 0;
		}
#endif
		init_worker_context(workerid);

		_STARPU_TRACE_WORKER_INIT_END(workerid);
	}
	{
		char thread_name[16];
		snprintf(thread_name, sizeof(thread_name), "CUDA %u", worker0->devid);
		starpu_pthread_setname(thread_name);
	}

	/* tell the main thread that this one is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker0->mutex);
	worker0->status = STATUS_UNKNOWN;
	worker0->worker_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker0->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker0->mutex);

	/* tell the main thread that this one is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
	worker_set->set_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker_set->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);

	return 0;
}

int _starpu_cuda_driver_run_once(struct _starpu_worker_set *worker_set)
{
	struct _starpu_worker *worker0 = &worker_set->workers[0];
	unsigned memnode = worker0->memory_node;
	struct starpu_task *tasks[worker_set->nworkers], *task;
	struct _starpu_job *j;
	int i, res;

	int idle;

	/* First poll for completed jobs */
	idle = 0;
	for (i = 0; i < (int) worker_set->nworkers; i++)
	{
		struct _starpu_worker *worker = &worker_set->workers[i];
		int workerid = worker->workerid;

		if (!worker->ntasks)
		{
			idle++;
			/* Even nothing to test */
			continue;
		}

		if (worker->pipeline_length)
			task = worker->current_tasks[worker->first_task];
		else
			task = worker->current_task;

		/* On-going asynchronous task, check for its termination first */
#ifdef STARPU_SIMGRID
		if (task_finished[workerid][worker->first_task])
#else /* !STARPU_SIMGRID */
		cudaError_t cures = cudaEventQuery(task_events[workerid][worker->first_task]);

		if (cures != cudaSuccess)
		{
			STARPU_ASSERT_MSG(cures == cudaErrorNotReady, "CUDA error on task %p, codelet %p (%s): %s (%d)", task, task->cl, _starpu_codelet_get_model_name(task->cl), cudaGetErrorString(cures), cures);
		}
		else
#endif /* !STARPU_SIMGRID */
		{
			_STARPU_TRACE_END_PROGRESS(memnode);
			/* Asynchronous task completed! */
			_starpu_set_local_worker_key(worker);
			finish_job_on_cuda(_starpu_get_job_associated_to_task(task), worker);
			/* See next task if any */
			if (worker->ntasks)
			{
				task = worker->current_tasks[worker->first_task];
				j = _starpu_get_job_associated_to_task(task);
				if (task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC)
				{
					/* An asynchronous task, it was already
					 * queued, it's now running, record its start time.  */
					_starpu_driver_start_job(worker, j, &worker->perf_arch, 0, starpu_profiling_status_get());
				}
				else
				{
					/* A synchronous task, we have finished
					 * flushing the pipeline, we can now at
					 * last execute it.  */

					_STARPU_TRACE_EVENT("sync_task");
					execute_job_on_cuda(task, worker);
					_STARPU_TRACE_EVENT("end_sync_task");
					worker->pipeline_stuck = 0;
				}
			}
#ifdef STARPU_USE_FXT
			int k;
			for (k = 0; k < (int) worker_set->nworkers; k++)
				if (worker_set->workers[k].ntasks)
					break;
			if (k == (int) worker_set->nworkers)
				/* Everybody busy */
				_STARPU_TRACE_END_EXECUTING()
#endif
			_STARPU_TRACE_START_PROGRESS(memnode);
		}

		if (!worker->pipeline_length || worker->ntasks < worker->pipeline_length)
			idle++;
	}

#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (!idle)
	{
		/* Nothing ready yet, no better thing to do than waiting */
		__starpu_datawizard_progress(memnode, 1, 0);
		__starpu_datawizard_progress(STARPU_MAIN_RAM, 1, 0);
		return 0;
	}
#endif

	/* Something done, make some progress */
	__starpu_datawizard_progress(memnode, 1, 1);
	__starpu_datawizard_progress(STARPU_MAIN_RAM, 1, 1);

	/* And pull tasks */
	res = _starpu_get_multi_worker_task(worker_set->workers, tasks, worker_set->nworkers, worker0->memory_node);

	if (!res)
		return 0;

	for (i = 0; i < (int) worker_set->nworkers; i++)
	{
		struct _starpu_worker *worker = &worker_set->workers[i];

		task = tasks[i];
		if (!task)
			continue;


		j = _starpu_get_job_associated_to_task(task);

		/* can CUDA do that task ? */
		if (!_STARPU_CUDA_MAY_PERFORM(j))
		{
			/* this is neither a cuda or a cublas task */
			worker->ntasks--;
			_starpu_push_task_to_workers(task);
			continue;
		}

		if (worker->ntasks > 1 && !(task->cl->cuda_flags[j->nimpl] & STARPU_CUDA_ASYNC))
		{
			/* We have to execute a non-asynchronous task but we
			 * still have tasks in the pipeline...  Record it to
			 * prevent more tasks from coming, and do it later */
			worker->pipeline_stuck = 1;
			continue;
		}

		_starpu_set_local_worker_key(worker);

		_STARPU_TRACE_END_PROGRESS(memnode);
		execute_job_on_cuda(task, worker);
		_STARPU_TRACE_START_PROGRESS(memnode);
	}

	return 0;
}

int _starpu_cuda_driver_deinit(struct _starpu_worker_set *worker_set)
{
	int lastdevid = -1;
	unsigned i;
	_STARPU_TRACE_WORKER_DEINIT_START;

	for (i = 0; i < worker_set->nworkers; i++)
	{
		struct _starpu_worker *worker = &worker_set->workers[i];
		unsigned devid = worker->devid;
		unsigned memnode = worker->memory_node;
		if ((int) devid == lastdevid)
			/* Already initialized */
			continue;
		lastdevid = devid;

		_starpu_handle_all_pending_node_data_requests(memnode);

		/* In case there remains some memory that was automatically
		 * allocated by StarPU, we release it now. Note that data
		 * coherency is not maintained anymore at that point ! */
		_starpu_free_all_automatically_allocated_buffers(memnode);

		_starpu_malloc_shutdown(memnode);

#ifndef STARPU_SIMGRID
		deinit_device_context(devid);
#endif /* !STARPU_SIMGRID */
	}

	for (i = 0; i < worker_set->nworkers; i++)
	{
		unsigned workerid = worker_set->workers[i].workerid;

		deinit_worker_context(workerid);
	}

	worker_set->workers[0].worker_is_initialized = 0;
	_STARPU_TRACE_WORKER_DEINIT_END(_STARPU_FUT_CUDA_KEY);

	return 0;
}

void *_starpu_cuda_worker(void *_arg)
{
	struct _starpu_worker_set* worker_set = _arg;
	unsigned i;

	_starpu_cuda_driver_init(worker_set);
	for (i = 0; i < worker_set->nworkers; i++)
		_STARPU_TRACE_START_PROGRESS(worker_set->workers[i].memory_node);
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_cuda_driver_run_once(worker_set);
	}
	for (i = 0; i < worker_set->nworkers; i++)
		_STARPU_TRACE_END_PROGRESS(worker_set->workers[i].memory_node);
	_starpu_cuda_driver_deinit(worker_set);

	return NULL;
}

#ifdef STARPU_USE_CUDA
void starpu_cublas_report_error(const char *func, const char *file, int line, int status)
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
	_STARPU_MSG("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ABORT();
}

void starpu_cuda_report_error(const char *func, const char *file, int line, cudaError_t status)
{
	const char *errormsg = cudaGetErrorString(status);
	printf("oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ABORT();
}
#endif /* STARPU_USE_CUDA */

#ifdef STARPU_USE_CUDA
int
starpu_cuda_copy_async_sync(void *src_ptr, unsigned src_node,
			    void *dst_ptr, unsigned dst_node,
			    size_t ssize, cudaStream_t stream,
			    enum cudaMemcpyKind kind)
{
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
	int peer_copy = 0;
	int src_dev = -1, dst_dev = -1;
#endif
	cudaError_t cures = 0;

	if (kind == cudaMemcpyDeviceToDevice && src_node != dst_node)
	{
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
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
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
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
#ifdef STARPU_HAVE_CUDA_MEMCPY_PEER
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

		if (!cures)
			cures = cudaDeviceSynchronize();
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		return 0;
	}

	return -EAGAIN;
}
#endif /* STARPU_USE_CUDA */

int _starpu_run_cuda(struct _starpu_worker_set *workerarg)
{
	/* Let's go ! */
	_starpu_cuda_worker(workerarg);

	return 0;
}
