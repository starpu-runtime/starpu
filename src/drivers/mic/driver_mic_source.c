/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Inria
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

#include <scif.h>

#include <starpu.h>
#include <starpu_profiling.h>
#include <core/sched_policy.h>
#include <core/workers.h>
#include <common/uthash.h>

#include <drivers/driver_common/driver_common.h>
#include <drivers/mp_common/source_common.h>

#include "driver_mic_common.h"
#include "driver_mic_source.h"


/* Array of structures containing all the informations useful to send
 * and receive informations with devices */
struct _starpu_mp_node *mic_nodes[STARPU_MAXMICDEVS];

static COIENGINE handles[STARPU_MAXMICDEVS];

/* Structure used by host to store informations about a kernel executable on
 * a MIC device : its name, and its address on each device.
 * If a kernel has been initialized, then a lookup has already been achieved and the
 * device knows how to call it, else the host still needs to do a lookup.
 */
struct _starpu_mic_kernel
{
	UT_hash_handle hh;
	char *name;
	starpu_mic_kernel_t func[STARPU_MAXMICDEVS];
} *kernels;

/* Mutex for concurrent access to the table.
 */
starpu_pthread_mutex_t htbl_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Number of MIC worker initialized.
 */
unsigned int nb_mic_worker_init = 0;
starpu_pthread_mutex_t nb_mic_worker_init_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Returns the ID of the MIC device controlled by the caller.
 * if the worker doesn't control a MIC device -ENODEV is returned
 */

//static int _starpu_mic_get_devid(void)
//{
//	struct _starpu_machine_config *config = _starpu_get_machine_config();
//	int workerid = starpu_worker_get_id();
//
//	if (config->workers[workerid].arch != STARPU_MIC_WORKER)
//		return -ENODEV;
//
//	return config->workers[workerid].devid;
//}

struct _starpu_mp_node *_starpu_mic_src_get_actual_thread_mp_node()
{
	struct _starpu_worker *actual_worker = _starpu_get_local_worker_key();
	STARPU_ASSERT(actual_worker);

	int nodeid = actual_worker->mp_nodeid;
	STARPU_ASSERT(nodeid >= 0 && nodeid < STARPU_MAXMICDEVS);

	return mic_nodes[nodeid];
}

const struct _starpu_mp_node *_starpu_mic_src_get_mp_node_from_memory_node(int memory_node)
{
	int nodeid = _starpu_memory_node_get_devid(memory_node);
	STARPU_ASSERT_MSG(nodeid >= 0 && nodeid < STARPU_MAXMICDEVS, "bogus nodeid %d for memory node %d\n", nodeid, memory_node);

	return mic_nodes[nodeid];
}

static void _starpu_mic_src_free_kernel(void *kernel)
{
	struct _starpu_mic_kernel *k = kernel;

	free(k->name);
	free(kernel);
}

void _starpu_mic_clear_kernels(void)
{
	struct _starpu_mic_kernel *kernel, *tmp;
	HASH_ITER(hh, kernels, kernel, tmp)
	{
		HASH_DEL(kernels, kernel);
		free(kernel);
	}
}

int _starpu_mic_src_register_kernel(starpu_mic_func_symbol_t *symbol, const char *func_name)
{
	unsigned int func_name_size = (strlen(func_name) + 1) * sizeof(char);

	STARPU_PTHREAD_MUTEX_LOCK(&htbl_mutex);
	struct _starpu_mic_kernel *kernel;
	
	HASH_FIND_STR(kernels, func_name, kernel);

	if (kernel != NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		// Function already in the table.
		*symbol = kernel;
		return 0;
	}

	kernel = malloc(sizeof(*kernel));
	if (kernel == NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		return -ENOMEM;
	}

	kernel->name = malloc(func_name_size);
	if (kernel->name == NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		free(kernel);
		return -ENOMEM;
	}

	memcpy(kernel->name, func_name, func_name_size);

	HASH_ADD_STR(kernels, name, kernel);

	unsigned int nb_mic_devices = _starpu_mic_src_get_device_count();
	unsigned int i;
	for (i = 0; i < nb_mic_devices; ++i)
		kernel->func[i] = NULL;

	STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);

	*symbol = kernel;

	return 0;
}


starpu_mic_kernel_t _starpu_mic_src_get_kernel(starpu_mic_func_symbol_t symbol)
{
	int workerid = starpu_worker_get_id();
	
	/* This function has to be called in the codelet only, by the thread
	 * which will handle the task */
	if (workerid < 0)
		return NULL;

	int nodeid = starpu_worker_get_mp_nodeid(workerid);

	struct _starpu_mic_kernel *kernel = symbol;

	if (kernel->func[nodeid] == NULL)
	{
		struct _starpu_mp_node *node = mic_nodes[nodeid];
		int ret = _starpu_src_common_lookup(node, (void (**)(void))&kernel->func[nodeid], kernel->name);
		if (ret)
			return NULL;
	}

	return kernel->func[nodeid];
}

/* Report an error which occured when using a MIC device
 * and print this error in a human-readable style.
 * It hanbles errors occuring when using COI.
 */

void _starpu_mic_src_report_coi_error(const char *func, const char *file,
				      const int line, const COIRESULT status)
{
	const char *errormsg = COIResultGetName(status);
	printf("SRC: oops in %s (%s:%u)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ASSERT(0);
}

/* Report an error which occured when using a MIC device
 * and print this error in a human-readable style.
 * It hanbles errors occuring when using SCIF.
 */

void _starpu_mic_src_report_scif_error(const char *func, const char *file, const int line, const int status)
{
	const char *errormsg = strerror(status);
	printf("SRC: oops in %s (%s:%u)... %d: %s \n", func, file, line, status, errormsg);
	STARPU_ASSERT(0);
}

/* Return the number of MIC devices in the system.
 * If the number of devices is already known, we use the cached value
 * without calling again COI. */

unsigned _starpu_mic_src_get_device_count(void)
{
	static unsigned short cached = 0;
	static unsigned nb_devices = 0;

	/* We don't need to call the COI API again if we already
	 * have the result in cache */
	if (!cached)
	{
		COIRESULT res;
		res = COIEngineGetCount(COI_ISA_MIC, &nb_devices);

		/* If something is wrong with the COI engine, we shouldn't
		 * use MIC devices (if there is any...) */
		if (res != COI_SUCCESS)
			nb_devices = 0;

		cached = 1;
	}

	return nb_devices;
}

unsigned starpu_mic_device_get_count(void)
{
    // Return the number of configured MIC devices.
    struct _starpu_machine_config *config = _starpu_get_machine_config ();
    struct _starpu_machine_topology *topology = &config->topology;

    return topology->nmicdevices;
}

starpu_mic_kernel_t _starpu_mic_src_get_kernel_from_codelet(struct starpu_codelet *cl, unsigned nimpl)
{
	starpu_mic_kernel_t kernel = NULL;

	starpu_mic_func_t func = _starpu_task_get_mic_nth_implementation(cl, nimpl);
	if (func)
	{
		/* We execute the function contained in the codelet, it must return a
		 * pointer to the function to execute on the device, either specified
		 * directly by the user or by a call to starpu_mic_get_func().
		 */
		kernel = func();
	}
	else
	{
		/* If user dont define any starpu_mic_fun_t in cl->mic_func we try to use
		 * cpu_func_name.
		 */
		char *func_name = _starpu_task_get_cpu_name_nth_implementation(cl, nimpl);
		if (func_name)
		{
			starpu_mic_func_symbol_t symbol;

			_starpu_mic_src_register_kernel(&symbol, func_name);

			kernel = _starpu_mic_src_get_kernel(symbol);
		}
	}
	STARPU_ASSERT(kernel);

	return kernel;
}



void(* _starpu_mic_src_get_kernel_from_job(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *j))(void)
{
	starpu_mic_kernel_t kernel = NULL;

	starpu_mic_func_t func = _starpu_task_get_mic_nth_implementation(j->task->cl, j->nimpl);
	if (func)
	{
		/* We execute the function contained in the codelet, it must return a
		 * pointer to the function to execute on the device, either specified
		 * directly by the user or by a call to starpu_mic_get_func().
		 */
		kernel = func();
	}
	else
	{
		/* If user dont define any starpu_mic_fun_t in cl->mic_func we try to use
		 * cpu_func_name.
		 */
		char *func_name = _starpu_task_get_cpu_name_nth_implementation(j->task->cl, j->nimpl);
		if (func_name)
		{
			starpu_mic_func_symbol_t symbol;

			_starpu_mic_src_register_kernel(&symbol, func_name);

			kernel = _starpu_mic_src_get_kernel(symbol);
		}
	}
	STARPU_ASSERT(kernel);

	return (void (*)(void))kernel;
}



/* Initialize the node structure describing the MIC source.
 */
void _starpu_mic_src_init(struct _starpu_mp_node *node)
{
	/* Let's initialize the connection with the peered sink device */
	_starpu_mic_common_connect(&node->mp_connection.mic_endpoint,
					    STARPU_TO_MIC_ID(node->peer_id),
					    STARPU_MIC_SINK_PORT_NUMBER(node->peer_id),
					    STARPU_MIC_SOURCE_PORT_NUMBER);

	_starpu_mic_common_connect(&node->host_sink_dt_connection.mic_endpoint,
				   STARPU_TO_MIC_ID(node->peer_id),
				   STARPU_MIC_SINK_DT_PORT_NUMBER(node->peer_id),
				   STARPU_MIC_SOURCE_DT_PORT_NUMBER);
}

/* Deinitialize the MIC sink, close all the connections.
 */
void _starpu_mic_src_deinit(struct _starpu_mp_node *node)
{
	scif_close(node->host_sink_dt_connection.mic_endpoint);
	scif_close(node->mp_connection.mic_endpoint);
}

/* Get infos of the MIC associed to memory_node */
static void _starpu_mic_get_engine_info(COI_ENGINE_INFO *info, int devid)
{
	STARPU_ASSERT(devid >= 0 && devid < STARPU_MAXMICDEVS);

	if (COIEngineGetInfo(handles[devid], sizeof(*info), info) != COI_SUCCESS)
		STARPU_MIC_SRC_REPORT_COI_ERROR(errno);
}

/* TODO: call _starpu_memory_manager_set_global_memory_size instead */
/* Return the size of the memory on the MIC associed to memory_node */
size_t _starpu_mic_get_global_mem_size(int devid)
{
	COI_ENGINE_INFO infos;
	_starpu_mic_get_engine_info(&infos, devid);

	return infos.PhysicalMemory;
}

/* Return the size of the free memory on the MIC associed to memory_node */
size_t _starpu_mic_get_free_mem_size(int devid)
{
	COI_ENGINE_INFO infos;
	_starpu_mic_get_engine_info(&infos, devid);

	return infos.PhysicalMemoryFree;
}

/* Allocate memory on MIC.
 * Return 0 if OK or 1 if not.
 */
int _starpu_mic_allocate_memory(void **addr, size_t size, unsigned memory_node)
{
	/* We check we have (1.25 * size) free space in the MIC because
	 * transfert with scif is not possible when the MIC
	 * doesn't have enought free memory.
	 * In this cas we can't tell any things to the host. */
	//int devid = _starpu_memory_node_get_devid(memory_node);
	//if (_starpu_mic_get_free_mem_size(devid) < size * 1.25)
	//	return 1;

	const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(memory_node);

	return _starpu_src_common_allocate(mp_node, addr, size);
}

/* Free memory on MIC.
 * Mic need size to free memory for use the function scif_unregister.
 */
void _starpu_mic_free_memory(void *addr, size_t size, unsigned memory_node)
{
	const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(memory_node);
	struct _starpu_mic_free_command cmd = {addr, size};

	return _starpu_mp_common_send_command(mp_node, STARPU_FREE, &cmd, sizeof(cmd));
}

/* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 * node to the address pointed by DST in the DST_NODE memory node
 */
int _starpu_mic_copy_ram_to_mic(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size)
{
	const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(dst_node);

	return _starpu_src_common_copy_host_to_sink(mp_node, src, dst, size);
}

/* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 * node to the address pointed by DST in the DST_NODE memory node
 */
int _starpu_mic_copy_mic_to_ram(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size)
{
	const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(src_node);

	return _starpu_src_common_copy_sink_to_host(mp_node, src, dst, size);
}

/* Asynchronous transfers */
int _starpu_mic_copy_ram_to_mic_async(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size)
{
	const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(dst_node);

	if (scif_vwriteto(mp_node->host_sink_dt_connection.mic_endpoint, src, size, (off_t)dst, 0) < 0)
		STARPU_MIC_SRC_REPORT_SCIF_ERROR(errno);

	return 0;
}

int _starpu_mic_copy_mic_to_ram_async(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size)
{
	const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(src_node);

	if (scif_vreadfrom(mp_node->host_sink_dt_connection.mic_endpoint, dst, size, (off_t)src, 0) < 0)
		STARPU_MIC_SRC_REPORT_SCIF_ERROR(errno);

	return 0;
}

/* Initialize a _starpu_mic_async_event. */
int _starpu_mic_init_event(struct _starpu_mic_async_event *event, unsigned memory_node)
{
	const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(memory_node);
	scif_epd_t epd = mp_node->host_sink_dt_connection.mic_endpoint;

	event->memory_node = memory_node;

	/* Address of allocation must be multiple of the page size. */
	if (posix_memalign((void **)&(event->signal), 0x1000, sizeof(*(event->signal))) != 0)
		return -ENOMEM;
	*(event->signal) = 0;

	/* The size pass to scif_register is 0x1000 because it should be a multiple of the page size. */
	if (scif_register(epd, event->signal, 0x1000, (off_t)(event->signal), SCIF_PROT_WRITE, SCIF_MAP_FIXED) < 0)
		STARPU_MIC_SRC_REPORT_SCIF_ERROR(errno);

	/* Mark for a futur wait. */
	if (scif_fence_mark(epd, SCIF_FENCE_INIT_SELF, &(event->mark)) < 0)
		STARPU_MIC_SRC_REPORT_SCIF_ERROR(errno);

	/* Tell to scif to write STARPU_MIC_REQUEST_COMPLETE in event->signal when the transfer is complete.
	 * We use this for test the end of a transfer. */
	if (scif_fence_signal(epd, (off_t)event->signal, STARPU_MIC_REQUEST_COMPLETE, 0, 0, SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_LOCAL) < 0)
		STARPU_MIC_SRC_REPORT_SCIF_ERROR(errno);

	return 0;
}

/* Wait the end of the asynchronous request */
void _starpu_mic_wait_request_completion(struct _starpu_mic_async_event *event)
{
	if (event->signal != NULL)
	{
		const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(event->memory_node);
		scif_epd_t epd = mp_node->host_sink_dt_connection.mic_endpoint;

		if (scif_fence_wait(epd, event->mark) < 0)
			STARPU_MIC_SRC_REPORT_SCIF_ERROR(errno);

		if (scif_unregister(epd, (off_t)(event->signal), 0x1000) < 0)
			STARPU_MIC_SRC_REPORT_SCIF_ERROR(errno);

		free(event->signal);
		event->signal = NULL;
	}
}

/* Test if a asynchronous request is end.
 * Return 1 if is end, 0 else. */
int _starpu_mic_request_is_complete(struct _starpu_mic_async_event *event)
{
	if (event->signal != NULL && *(event->signal) != STARPU_MIC_REQUEST_COMPLETE)
		return 0;

	const struct _starpu_mp_node *mp_node = _starpu_mic_src_get_mp_node_from_memory_node(event->memory_node);
	scif_epd_t epd = mp_node->host_sink_dt_connection.mic_endpoint;

	if (scif_unregister(epd, (off_t)(event->signal), 0x1000) < 0)
		STARPU_MIC_SRC_REPORT_SCIF_ERROR(errno);

	free(event->signal);
	event->signal = NULL;
	return 1;
}



void *_starpu_mic_src_worker(void *arg)
{
	struct _starpu_worker_set *worker_set = arg;
	/* As all workers of a set share common data, we just use the first
	 * one for intializing the following stuffs. */
	struct _starpu_worker *baseworker = &worker_set->workers[0];
	struct _starpu_machine_config *config = baseworker->config;
	unsigned baseworkerid = baseworker - config->workers;
	unsigned mp_nodeid = baseworker->mp_nodeid;
	unsigned i;

	/* unsigned memnode = baseworker->memory_node; */

	_starpu_worker_init(baseworker, _STARPU_FUT_MIC_KEY);

	// Current task for a thread managing a worker set has no sense.
	_starpu_set_current_task(NULL);

	for (i = 0; i < config->topology.nmiccores[mp_nodeid]; i++)
	{
		struct _starpu_worker *worker = &config->workers[baseworkerid+i];
		snprintf(worker->name, sizeof(worker->name), "MIC %d core %u", mp_nodeid, i);
	}

	baseworker->status = STATUS_UNKNOWN;

	_STARPU_TRACE_WORKER_INIT_END;

	/* tell the main thread that this one is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
	worker_set->set_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker_set->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);

	_starpu_src_common_worker(worker_set, baseworkerid, mic_nodes[mp_nodeid]);

	_STARPU_TRACE_WORKER_DEINIT_START;

	_STARPU_TRACE_WORKER_DEINIT_END(_STARPU_FUT_CUDA_KEY);

	return NULL;

}
