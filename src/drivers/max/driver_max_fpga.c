/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_max_fpga.h>
#include <starpu_profiling.h>
#include <common/utils.h>
#include <common/config.h>
#include <core/debug.h>
#include <core/devices.h>
#include <drivers/driver_common/driver_common.h>
#include "driver_max_fpga.h"
#include <core/sched_policy.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>

/* the number of FPGA devices */
static unsigned	nmax_fpga;
static size_t max_fpga_mem[STARPU_MAXMAXFPGADEVS];
static max_engine_t *engines[STARPU_MAXMAXFPGADEVS];
static fpga_mem current_address[STARPU_MAXMAXFPGADEVS];

static unsigned max_fpga_bindid_init[STARPU_MAXMAXFPGADEVS];
static unsigned max_fpga_bindid[STARPU_MAXMAXFPGADEVS];
static unsigned max_fpga_memory_init[STARPU_MAXMAXFPGADEVS];
static unsigned max_fpga_memory_nodes[STARPU_MAXMAXFPGADEVS];

static void _starpu_max_fpga_limit_max_fpga_mem(unsigned );
static size_t _starpu_max_fpga_get_max_fpga_mem_size(unsigned devid);

static size_t _starpu_max_fpga_get_max_fpga_mem_size(unsigned devid)
{
	return max_fpga_mem[devid];
}

max_engine_t *starpu_max_fpga_get_local_engine(void)
{
	int worker = starpu_worker_get_id_check();
	int devid = starpu_worker_get_devid(worker);

	STARPU_ASSERT_MSG(engines[devid], "engine for fpga %d on worker %d is NULL!?", devid, worker);

	return engines[devid];
}

/* This is called to initialize FPGA and discover devices */
void _starpu_init_max_fpga()
{
	memset(&max_fpga_bindid_init, 0, sizeof(max_fpga_bindid_init));
	memset(&max_fpga_memory_init, 0, sizeof(max_fpga_memory_init));
}

static void _starpu_initialize_workers_max_fpga_deviceid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

	_starpu_initialize_workers_deviceid(uconf->use_explicit_workers_max_fpga_deviceid == 0
					    ? NULL
					    : (int *)uconf->workers_max_fpga_deviceid,
					    &(config->current_devid[STARPU_MAX_FPGA_WORKER]),
					    (int *)topology->workers_max_fpga_deviceid,
					    "STARPU_WORKERS_MAX_FPGAID",
					    topology->nhwdevices[STARPU_MAX_FPGA_WORKER],
					    STARPU_MAX_FPGA_WORKER);
	_starpu_devices_drop_duplicate(topology->workers_max_fpga_deviceid);
}

unsigned _starpu_max_fpga_get_device_count(void)
{
	return nmax_fpga;
}

/* This is called to really discover the hardware */
void _starpu_max_fpga_discover_devices (struct _starpu_machine_config *config)
{
	//TODO: This is statically assigned, in the next round of integration
	// I will have to read from the struct fpga in fpga
	struct starpu_max_load *load = _starpu_config.conf.max_fpga_load;
	const char *sim_socket = max_config_get_string(MAX_CONFIG_USE_SIMULATION);
	int n;

	if (!load)
	{
		/* Nothing specified, single-FPGA execution with basic static
		 * interface, file will be auto-loaded by SLiC. */
		n = 1;
	}
	else
	{
		struct starpu_max_load *cur, *star = NULL;
		size_t nstar = 0;

		/* First check if we have a star, we will want to subtract non-star loads from it */
		for (cur =  load; cur->engine_id_pattern; cur++)
			if (!strcmp(cur->engine_id_pattern, "*")
			 ||  strstr(cur->engine_id_pattern, ":*"))
			{
				STARPU_ASSERT_MSG(!cur[1].file, "in starpu_max_load array, * pattern must be last");
				star = cur;

				if (sim_socket)
					/* not specified, assume 1 */
					nstar = 1;
				else
					nstar = max_count_engines_free(cur->file, star->engine_id_pattern);
				break;
			}

		n = 0;
		/* Now check the non-star loads */
		for (cur = load; cur != star && cur->engine_id_pattern; cur++)
		{
			size_t size;

			size = max_count_engines_free(load->file, load->engine_id_pattern);
			STARPU_ASSERT_MSG(size > 0, "cannot load starpu_max_load element %u on %s", (unsigned) (cur - load), load->engine_id_pattern);
			/* One FPGA more to be used */
			n++;

			if (star)
			{
				size = max_count_engines_free(load->file, star->engine_id_pattern);
				if (size > 1)
					/* One of the star devices will be used to load this file */
					nstar--;
			}
		}
		n += nstar;
	}

	//LMemInterface addLMemInterface()
	//// pour récupérer l'accès à la LMem

	if (n > STARPU_MAXMAXFPGADEVS)
	{
		_STARPU_DISP("Warning: %d Maxeler FPGA devices available. Only %d enabled. Use configure option --enable-maxmaxfpgadev=xxx to update the maximum value of supported Maxeler FPGA devices.\n", n, STARPU_MAXMAXFPGADEVS);
		n = STARPU_MAXMAXFPGADEVS;
	}

	config->topology.nhwdevices[STARPU_MAX_FPGA_WORKER] = nmax_fpga = n;
}

/* Determine which devices we will use */
void _starpu_init_max_fpga_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *)
{
	int nmax_fpga = config->conf.nmax_fpga;
	if (nmax_fpga != 0)
	{
		/* The user did not disable FPGA. We need to initialize
		 * FPGA early to count the number of devices */
		_starpu_init_max_fpga();
		int nb_devices = _starpu_max_fpga_get_device_count();

		_starpu_topology_check_ndevices(&nmax_fpga, nb_devices, 0, STARPU_MAXMAXFPGADEVS, "nmax_fpga", "Maxeler FPGA", "maxmaxfpgadev");
	}

	/* Now we know how many MAX FPGA devices will be used */
	topology->ndevices[STARPU_MAX_FPGA_WORKER] = nmax_fpga;

	_starpu_initialize_workers_max_fpga_deviceid(config);

	unsigned max_fpga;
	for (max_fpga = 0; (int) max_fpga < nmax_fpga; max_fpga++)
	{
		int devid = _starpu_get_next_devid(topology, config, STARPU_MAX_FPGA_WORKER);
		if (devid == -1)
		{
			// There is no more devices left
			topology->ndevices[STARPU_MAX_FPGA_WORKER] = max_fpga;
			break;
		}

		_starpu_topology_configure_workers(topology, config,
				STARPU_MAX_FPGA_WORKER,
				max_fpga, devid, 0, 0,
				1, 1, NULL, NULL);
	}
}

/* Bind the driver on a CPU core */
void _starpu_max_fpga_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned *preferred_binding = NULL;
	unsigned npreferred = 0;

	if (max_fpga_bindid_init[devid])
	{
		workerarg->bindid = max_fpga_bindid[devid];
	}
	else
	{
		max_fpga_bindid_init[devid] = 1;
		workerarg->bindid = max_fpga_bindid[devid] = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, preferred_binding, npreferred);
	}
}

/* Set up memory and buses */
void _starpu_max_fpga_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned memory_node = -1;
	/* Perhaps the worker has some "favourite" bindings  */
	unsigned devid = workerarg->devid;
	unsigned numa;

	if (max_fpga_memory_init[devid])
	{
		memory_node = max_fpga_memory_nodes[devid];
	}
	else
	{
		max_fpga_memory_init[devid] = 1;

		memory_node = max_fpga_memory_nodes[devid] = _starpu_memory_node_register(STARPU_MAX_FPGA_RAM, devid);
		_starpu_register_bus(STARPU_MAIN_RAM, memory_node);
		_starpu_register_bus(memory_node, STARPU_MAIN_RAM);

	}
	_starpu_memory_node_add_nworkers(memory_node);

	//This worker can manage transfers on NUMA nodes
	for (numa = 0; numa < starpu_memory_nodes_get_numa_count(); numa++)
			_starpu_worker_drives_memory_node(workerarg, numa);

	_starpu_worker_drives_memory_node(workerarg, memory_node);

	workerarg->memory_node = memory_node;
}

static void _starpu_max_fpga_limit_max_fpga_mem(unsigned devid)
{
	starpu_ssize_t limit=-1;

	//TODO
	limit = starpu_get_env_number("STARPU_LIMIT_MAX_FPGA_MEM");
	if(limit != -1)
		max_fpga_mem[devid] = limit*1024*1024;
}

static void init_device_context(unsigned devid)
{
	struct starpu_max_load *load = _starpu_config.conf.max_fpga_load;

	/* 0 would be seen as NULL, i.e. allocation failed... */
	// FIXME: Maxeler FPGAs want 192-byte alignment
	// TODO: use int	max_get_burst_size (max_file_t *maxfile, const char *name)
	current_address[devid] = (fpga_mem) (8192*192);
	max_fpga_mem[devid] = 128ULL*1024*1024*1024;

	_starpu_max_fpga_limit_max_fpga_mem(devid);

	if (!load)
	{
		/* Nothing specified, single-FPGA execution with basic static
		 * interface, file will be auto-loaded by SLiC. */
		return;
	}
	else
	{
		unsigned n;

		/* Which load we shall use */
		for (n = 0; load->file; n++, load++)
		{
			if (!strcmp(load->engine_id_pattern, "*")
			 ||  strstr(load->engine_id_pattern, ":*"))
				break;
			if (n == devid)
				break;
		}

		STARPU_ASSERT(load->file);

		if (!strcmp(load->engine_id_pattern, "*")
		  || strstr(load->engine_id_pattern, ":*"))
		{
			char s[strlen(load->engine_id_pattern) + 32];
			if (!strcmp(load->engine_id_pattern, "*"))
				snprintf(s, sizeof(s), "*:%u", (unsigned) devid);
			else
			{
				char *colon = strstr(load->engine_id_pattern, ":*");
				snprintf(s, sizeof(s), "%.*s:%u",
						(int) (colon - load->engine_id_pattern),
						load->engine_id_pattern,
						(unsigned) devid);
			}
			/* FIXME: this assumes that the loads are in-order.
			 * Ideally we'd detect which ones had an explicit load */
			engines[devid] = max_load(load->file, s);
			STARPU_ASSERT_MSG(engines[devid], "engine %u (part of *) could not be loaded\n", n);
		}
		else
		{
			engines[n] = max_load(load->file, load->engine_id_pattern);
			STARPU_ASSERT_MSG(engines[n], "engine %u could not be loaded\n", n);
		}
	}
}

int _starpu_max_fpga_driver_init(struct _starpu_worker *worker)
{
	int devid = worker->devid;
	_starpu_driver_start(worker, STARPU_MAX_FPGA_WORKER, 1);
	/* FIXME: when we have NUMA support, properly turn node number into NUMA node number */
	// TODO: drop test when we allocated a memory node for fpga
	if (worker->memory_node != STARPU_MAIN_RAM)
		_starpu_memory_manager_set_global_memory_size(worker->memory_node, _starpu_max_fpga_get_max_fpga_mem_size(worker->devid));

	// TODO: multiple fpga in same thread
	init_device_context(devid);

	snprintf(worker->name, sizeof(worker->name), "FPGA %d", devid);
	snprintf(worker->short_name, sizeof(worker->short_name), "FPGA %d", devid);
	starpu_pthread_setname(worker->short_name);

	_STARPU_TRACE_WORKER_INIT_END(worker->workerid);

	/* tell the main thread that we are ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
	worker->status = STATUS_UNKNOWN;
	worker->worker_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);
	return 0;
}

int _starpu_max_fpga_driver_deinit(struct _starpu_worker *fpga_worker)
{
	_STARPU_TRACE_WORKER_DEINIT_START;

	unsigned memnode = fpga_worker->memory_node;
	_starpu_datawizard_handle_all_pending_node_data_requests(memnode);

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	fpga_worker->worker_is_initialized = 0;
	_STARPU_TRACE_WORKER_DEINIT_END(STARPU_MAX_FPGA_WORKER);

	return 0;
}

static uintptr_t _starpu_max_fpga_allocate_memory(unsigned dst_node, size_t size, int flags)
{
	(void) flags;
	unsigned devid = starpu_memory_node_get_devid(dst_node);

	fpga_mem addr, next_addr;
	addr = current_address[devid];
	next_addr = current_address[devid] + size;
	if (next_addr >= (fpga_mem) max_fpga_mem[devid])
	{
		printf("Memory overflow on %u\n", devid);
		return 0;
	}
	current_address[devid] = next_addr;
	printf("fpga mem returned from allocation @: %p - %p\n",addr, addr + size);
	return (uintptr_t) addr;
}

static int _starpu_max_fpga_copy_ram_to_max_fpga(void *src, void *dst, size_t size)
{
	printf("ram to fpga, fpga @= %p\n",dst);
	memcpy(dst,src,size);
	return 0;
	// LMemLoopback_writeLMem(dst, size, src);
}

/* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 *  * node to the address pointed by DST in the DST_NODE memory node
 *   */
void copy_ram_to_max_fpga(void *src, void *dst, size_t size)
{
	(void) src; (void) dst; (void) size;
	printf("ram to fpga, fpga @= %p\n",dst);
	// LMemLoopback_writeLMem(size, dst, src);
}

void copy_max_fpga_to_ram(void *src, void *dst, size_t size)
{
	(void) src; (void) dst; (void) size;
	printf("ram to fpga, fpga @= %p\n",src);
	//LMemLoopback_readLMem(size, src, dst);
}

/* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 * node to the address pointed by DST in the DST_NODE memory node
 */
int _starpu_max_fpga_copy_max_fpga_to_ram(void *src, void *dst, size_t size)
{
	printf("fpga to ram, fpga @= %p\n",src);
	memcpy(dst,src,size);
	return 0;
	//LMemLoopback_readLMem(src, size, dst);
}

/* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 * node to the address pointed by DST in the DST_NODE memory node
 */
int _starpu_max_fpga_copy_fpga_to_fpga(void *src, void *dst, size_t size)
{
	printf("fpga to ram, fpga @= %p\n",src);
	memcpy(dst,src,size);
	return 0;
	//LMemLoopback_XXXLMem(src, size, dst);
}

/* Asynchronous transfers */
int _starpu_max_fpga_copy_ram_to_max_fpga_async(void *src, void *dst, size_t size)
{
	printf("ram to fpga, fpga @= %p\n",dst);
	memcpy(dst,src,size);
	return 0;
	// Trouver dans la doc une version asynchrone de LMemLoopback_writeLMem();
}

int _starpu_max_fpga_copy_max_fpga_to_ram_async(void *src, void *dst, size_t size)
{
	printf("fpga to ram, fpga @= %p\n",src);
	memcpy(dst,src,size);
	return 0;
}

int _starpu_run_fpga(struct _starpu_worker *workerarg)
{
	/* Let's go ! */
	_starpu_max_fpga_worker(workerarg);
	return 0;
}

int _starpu_max_fpga_copy_data_from_cpu_to_fpga(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t ssize, struct _starpu_async_channel *async_channel)
{
	return _starpu_max_fpga_copy_ram_to_max_fpga((char*) src + src_offset, (char*) dst + dst_offset, ssize);
}

int _starpu_max_fpga_copy_data_from_fpga_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t ssize, struct _starpu_async_channel *async_channel)
{
	return _starpu_max_fpga_copy_max_fpga_to_ram((char*) src + src_offset, (char*) dst + dst_offset, ssize);
}

int _starpu_max_fpga_copy_data_from_fpga_to_fpga(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t ssize, struct _starpu_async_channel *async_channel)
{
	return _starpu_max_fpga_copy_fpga_to_fpga((char*) src + src_offset, (char*) dst + dst_offset, ssize);
}

int _starpu_max_fpga_copy_interface_from_fpga_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);

	STARPU_ASSERT(src_kind == STARPU_MAX_FPGA_RAM && dst_kind == STARPU_CPU_RAM);

	int ret = 1;

	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_max_fpga_copy_disabled() ||
	    !(copy_methods->max_fpga_to_ram_async || copy_methods->any_to_any))
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->max_fpga_to_ram || copy_methods->any_to_any);
		if (copy_methods->max_fpga_to_ram)
			copy_methods->max_fpga_to_ram(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		//req->async_channel.type = STARPU_MAX_FPGA_RAM;
		if (copy_methods->max_fpga_to_ram_async)
			ret = copy_methods->max_fpga_to_ram_async(src_interface, src_node, dst_interface, dst_node);
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			ret = copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}
		//_starpu_max_fpga_init_event(&(req->async_channel.event.fpga_event), src_node);
	}
	return ret;
}

int _starpu_max_fpga_copy_interface_from_cpu_to_fpga(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_MAX_FPGA_RAM);

	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;

	if (!req || starpu_asynchronous_copy_disabled() || starpu_asynchronous_max_fpga_copy_disabled() ||
	    !(copy_methods->ram_to_max_fpga_async || copy_methods->any_to_any))
	{
		/* this is not associated to a request so it's synchronous */
		STARPU_ASSERT(copy_methods->ram_to_max_fpga || copy_methods->any_to_any);
		if (copy_methods->ram_to_max_fpga)
			copy_methods->ram_to_max_fpga(src_interface, src_node, dst_interface, dst_node);
		else
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, NULL);
	}
	else
	{
		//req->async_channel.type = STARPU_MAX_FPGA_RAM;
		if (copy_methods->ram_to_max_fpga_async)
			copy_methods->ram_to_max_fpga_async(src_interface, src_node, dst_interface, dst_node);
		else
		{
			STARPU_ASSERT(copy_methods->any_to_any);
			copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, &req->async_channel);
		}
		//_starpu_max_fpga_init_event(&(req->async_channel.event.fpga_event), dst_node);
	}
	return 0;
}

static int execute_job_on_fpga(struct _starpu_job *j, struct starpu_task *worker_task, struct _starpu_worker *fpga_args, int rank, struct starpu_perfmodel_arch* perf_arch)
{
	int ret;
	int profiling = starpu_profiling_status_get();

	struct starpu_task *task = worker_task;
	struct starpu_codelet *cl = task->cl;

	STARPU_ASSERT(cl);

	/* TODO: use asynchronous */
	ret = _starpu_fetch_task_input(task, j, 0);
	if (ret != 0)
	{
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return -EAGAIN;
	}

	/* Give profiling variable */
	_starpu_driver_start_job(fpga_args, j, perf_arch, rank, profiling);

	/* In case this is a Fork-join parallel task, the worker does not
	 * execute the kernel at all. */
	if ((rank == 0) || (cl->type != STARPU_FORKJOIN))
	{
		_starpu_cl_func_t func = _starpu_task_get_fpga_nth_implementation(cl, j->nimpl);

		STARPU_ASSERT_MSG(func, "when STARPU_MAX_FPGA is defined in 'where', fpga_func or max_fpga_funcs has to be defined");
		if (_starpu_get_disable_kernels() <= 0)
		{
			_STARPU_TRACE_START_EXECUTING();
			func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
			_STARPU_TRACE_END_EXECUTING();
		}
	}

	_starpu_driver_end_job(fpga_args, j, perf_arch, rank, profiling);

	_starpu_driver_update_job_feedback(j, fpga_args, perf_arch, profiling);

	_starpu_push_task_output(j);

	return 0;
}

int _starpu_max_fpga_driver_run_once(struct _starpu_worker *fpga_worker)
{
	unsigned memnode = fpga_worker->memory_node;
	int workerid = fpga_worker->workerid;

	_STARPU_TRACE_START_PROGRESS(memnode);
	_starpu_datawizard_progress(1);
	if (memnode != STARPU_MAIN_RAM)
	{
		_starpu_datawizard_progress(1);
	}
	_STARPU_TRACE_END_PROGRESS(memnode);

	struct _starpu_job *j;
	struct starpu_task *task;
	int res;

	task = _starpu_get_worker_task(fpga_worker, workerid, memnode);

	if (!task)
		return 0;

	j = _starpu_get_job_associated_to_task(task);

	/* can a cpu perform that task ? */
	if (!_STARPU_MAY_PERFORM(j, MAX_FPGA))
	{
		/* put it at the end of the queue ... XXX */
		_starpu_push_task_to_workers(task);
		return 0;
	}

	int rank = 0;
	int is_parallel_task = (j->task_size > 1);

	struct starpu_perfmodel_arch* perf_arch;

	if (is_parallel_task)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		rank = j->active_task_alias_count++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

		if(j->combined_workerid != -1)
		{
			struct _starpu_combined_worker *combined_worker;
			combined_worker = _starpu_get_combined_worker_struct(j->combined_workerid);

			fpga_worker->combined_workerid = j->combined_workerid;
			fpga_worker->worker_size = combined_worker->worker_size;
			fpga_worker->current_rank = rank;
			perf_arch = &combined_worker->perf_arch;
		}
		else
		{
			struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(fpga_worker, j);
			STARPU_ASSERT_MSG(sched_ctx != NULL, "there should be a worker %d in the ctx of this job \n", fpga_worker->workerid);

			perf_arch = &sched_ctx->perf_arch;
		}
	}
	else
	{
		fpga_worker->combined_workerid = fpga_worker->workerid;
		fpga_worker->worker_size = 1;
		fpga_worker->current_rank = 0;
		perf_arch = &fpga_worker->perf_arch;
	}

	_starpu_set_current_task(j->task);
	fpga_worker->current_task = j->task;
	j->workerid = fpga_worker->workerid;

	res = execute_job_on_fpga(j, task, fpga_worker, rank, perf_arch);

	_starpu_set_current_task(NULL);
	fpga_worker->current_task = NULL;

	if (res)
	{
		switch (res)
		{
		case -EAGAIN:
			_starpu_push_task_to_workers(task);
			return 0;
		default:
			STARPU_ABORT();
		}
	}

	/* In the case of combined workers, we need to inform the
	 * scheduler each worker's execution is over.
	 * Then we free the workers' task alias */
	if (is_parallel_task)
	{
		_starpu_sched_post_exec_hook(task);
		free(task);
	}

	if (rank == 0)
		_starpu_handle_job_termination(j);
	return 0;
}

void *_starpu_max_fpga_worker(void *_arg)
{
	struct _starpu_worker* worker = _arg;
	 unsigned memnode = worker->memory_node;

	_starpu_max_fpga_driver_init(worker);
	_STARPU_TRACE_START_PROGRESS(memnode);
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_max_fpga_driver_run_once(worker);
	}
	_STARPU_TRACE_END_PROGRESS(memnode);
	_starpu_max_fpga_driver_deinit(worker);

	return NULL;
}

struct _starpu_driver_ops _starpu_driver_max_fpga_ops =
{
	.init = _starpu_max_fpga_driver_init,
	.run = _starpu_run_fpga,
	.run_once = _starpu_max_fpga_driver_run_once,
	.deinit = _starpu_max_fpga_driver_deinit
};

// TODO: transfers
struct _starpu_node_ops _starpu_driver_max_fpga_node_ops =
{
	.name = "fpga driver",

	.malloc_on_node = _starpu_max_fpga_allocate_memory,
	.free_on_node = NULL,

	.is_direct_access_supported = NULL,

	//.copy_data_to[STARPU_CPU_RAM] = _starpu_max_fpga_copy_data_from_fpga_to_cpu,
	//.copy_data_to[STARPU_MAX_FPGA_RAM] = _starpu_max_fpga_copy_data_from_fpga_to_fpga,

	//.copy_data_from[STARPU_CPU_RAM] = _starpu_max_fpga_copy_data_from_cpu_to_fpga,
	//.copy_data_from[STARPU_MAX_FPGA_RAM] = _starpu_max_fpga_copy_data_from_fpga_to_fpga,

	//.copy_interface_to[STARPU_CPU_RAM] = _starpu_max_fpga_copy_interface_from_fpga_to_cpu,
	//.copy_interface_to[STARPU_MAX_FPGA_RAM] = _starpu_max_fpga_copy_interface_from_fpga_to_fpga,

	//.copy_interface_from[STARPU_CPU_RAM] = _starpu_max_fpga_copy_interface_from_cpu_to_fpga,
	//.copy_interface_from[STARPU_MAX_FPGA_RAM] = _starpu_max_fpga_copy_interface_from_fpga_to_fpga,

	.wait_request_completion = NULL,
	.test_request_completion = NULL,
};
