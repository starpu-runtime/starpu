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


#include <starpu.h>
#include <starpu_profiling.h>
#include <core/sched_policy.h>
#include <core/task.h>
#include <common/uthash.h>

#include <RCCE.h>

#include <drivers/driver_common/driver_common.h>
#include <drivers/mp_common/source_common.h>
#include <drivers/scc/driver_scc_common.h>
#include <drivers/scc/driver_scc_source.h>

static struct _starpu_mp_node *scc_mp_nodes[STARPU_MAXSCCDEVS];

struct _starpu_scc_kernel
{
	UT_hash_handle hh;
	char *name;
	starpu_scc_kernel_t func[STARPU_MAXSCCDEVS];
} *kernels;

starpu_pthread_mutex_t htbl_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

static struct _starpu_mp_node *_starpu_scc_src_memory_node_to_mp_node(unsigned memory_node)
{
	int devid = _starpu_memory_node_get_devid(memory_node);

	STARPU_ASSERT(devid < STARPU_MAXSCCDEVS);
	return scc_mp_nodes[devid];
}

static void _starpu_scc_src_init_context(int devid)
{
	/* Let's create the node structure, we'll communicate with the peer
	 * through RCCE thanks to it */
	scc_mp_nodes[devid] = _starpu_mp_common_node_create(STARPU_SCC_SOURCE, devid);
}

static void _starpu_scc_src_deinit_context(int devid)
{
	_starpu_mp_common_send_command(scc_mp_nodes[devid], STARPU_EXIT, NULL, 0);

	_starpu_mp_common_node_destroy(scc_mp_nodes[devid]);
}
void (*_starpu_scc_src_get_kernel_from_job(const struct _starpu_mp_node *,struct _starpu_job *j))(void)
{
  starpu_scc_kernel_t kernel = NULL;

  starpu_scc_func_t func = _starpu_task_get_scc_nth_implementation(j->task->cl, j->nimpl);
  if (func)
    {
      /* We execute the function contained in the codelet, it must return a
       * pointer to the function to execute on the device, either specified
       * directly by the user or by a call to starpu_scc_get_kernel().
       */
      kernel = func();
    }
  else
    {
      /* If user doesn't define any starpu_scc_func_t in cl->scc_funcs we try to use
       * cpu_funcs_name.
       */
      char *func_name = _starpu_task_get_cpu_name_nth_implementation(j->task->cl, j->nimpl);
      if (func_name)
	{
	  starpu_scc_func_symbol_t symbol;

	  _starpu_scc_src_register_kernel(&symbol, func_name);

	  kernel = _starpu_scc_src_get_kernel(symbol);
	}
    }
  STARPU_ASSERT(kernel);  

  return (void (*)(void))kernel;
}


void _starpu_scc_src_mp_deinit()
{
	_starpu_scc_common_unmap_shared_memory();
	RCCE_finalize();
}

int _starpu_scc_src_register_kernel(starpu_scc_func_symbol_t *symbol, const char *func_name)
{
	unsigned int func_name_size = (strlen(func_name) + 1) * sizeof(char);

	STARPU_PTHREAD_MUTEX_LOCK(&htbl_mutex);
	struct _starpu_scc_kernel *kernel;

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

	unsigned int nb_scc_devices = starpu_scc_worker_get_count();
	unsigned int i;
	for (i = 0; i < nb_scc_devices; ++i)
		kernel->func[i] = NULL;

	STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);

	*symbol = kernel;

	return 0;
}

starpu_scc_kernel_t _starpu_scc_src_get_kernel(starpu_scc_func_symbol_t symbol)
{
	int workerid = starpu_worker_get_id();
	/* This function has to be called in the codelet only, by the thread
	 * which will handle the task */
	if (workerid < 0)
		return NULL;

	int devid = starpu_worker_get_devid(workerid);

	struct _starpu_scc_kernel *kernel = symbol;

	if (kernel->func[devid] == NULL)
	{
		struct _starpu_mp_node *node = scc_mp_nodes[devid];
		int ret = _starpu_src_common_lookup(node, (void (**)(void))&kernel->func[devid], kernel->name);

		if (ret)
			return NULL;
	}

	return kernel->func[devid];
}

unsigned _starpu_scc_src_get_device_count()
{
	int nb_scc_devices;

	if (!_starpu_scc_common_is_mp_initialized())
	{
		return 0;
	}

	nb_scc_devices = RCCE_num_ues() - 1;
	nb_scc_devices = nb_scc_devices < 0 ? 0 : nb_scc_devices;

	return nb_scc_devices;
}

void _starpu_scc_exit_useless_node(int devid)
{
	struct _starpu_mp_node *node = _starpu_mp_common_node_create(STARPU_SCC_SOURCE, devid);

	_starpu_mp_common_send_command(node, STARPU_EXIT, NULL, 0);

	_starpu_mp_common_node_destroy(node);
}

void _starpu_scc_src_init(struct _starpu_mp_node *node)
{
	node->mp_connection.scc_nodeid = STARPU_TO_SCC_SINK_ID(node->peer_id);
}

/* Allocate memory on SCC.
 * Return 0 if OK or 1 if not.
 */
int _starpu_scc_allocate_memory(void **addr, size_t size, unsigned memory_node)
{
	return _starpu_src_common_allocate(_starpu_scc_src_memory_node_to_mp_node(memory_node),
			addr, size);
}

/* Free memory on SCC.
 */
void _starpu_scc_free_memory(void *addr, unsigned memory_node)
{
	return _starpu_src_common_free(_starpu_scc_src_memory_node_to_mp_node(memory_node),
			addr);
}

int _starpu_scc_allocate_shared_memory(void **addr, size_t size)
{
	return (*addr = (void*)RCCE_shmalloc(size)) == NULL;
}

void _starpu_scc_free_shared_memory(void *addr)
{
	RCCE_shfree(addr);
}

/* Assigns the offset to "offset" between "ptr" and the start of the shared memory.
 * Affect "dev_handle" with the start of the shared memory is useful for data
 * partionning.
 */
void _starpu_scc_set_offset_in_shared_memory(void *ptr, void **dev_handle, size_t *offset)
{
	/* We're on SCC... */
	if (_starpu_can_submit_scc_task())
	{
		if (!_starpu_scc_common_is_in_shared_memory(ptr))
		{
			fprintf(stderr, "The data (%p) you want to register does not seem to be allocated in shared memory. "
					"Please use starpu_malloc to do this.\n", ptr);
			STARPU_ABORT();
		}

		void *shm_addr = _starpu_scc_common_get_shared_memory_addr();

		if (dev_handle)
			*dev_handle = shm_addr;

		if (offset)
			*offset = ptr - shm_addr;
	}
}

/* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 * node to the address pointed by DST in the DST_NODE memory node
 */
int _starpu_scc_copy_src_to_sink(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size)
{
	return _starpu_src_common_copy_host_to_sink(_starpu_scc_src_memory_node_to_mp_node(dst_node),
			src, dst, size);
}

/* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
 * node to the address pointed by DST in the DST_NODE memory node
 */
int _starpu_scc_copy_sink_to_src(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size)
{
	return _starpu_src_common_copy_sink_to_host(_starpu_scc_src_memory_node_to_mp_node(src_node),
			src, dst, size);
}

int _starpu_scc_copy_sink_to_sink(void *src, unsigned src_node, void *dst, unsigned dst_node, size_t size)
{
	return _starpu_src_common_copy_sink_to_sink(_starpu_scc_src_memory_node_to_mp_node(src_node),
			_starpu_scc_src_memory_node_to_mp_node(dst_node),
			src, dst, size);
}

void *_starpu_scc_src_worker(void *arg)
{
	struct _starpu_worker_set *args = arg;

	int devid = args->devid;
	int workerid = args->workerid;
	struct _starpu_machine_config *config = args->config;
	unsigned memnode = args->memory_node;
	unsigned baseworkerid = args - config->workers;
	unsigned mp_nodeid = args->mp_nodeid;
	unsigned i;

	_starpu_worker_start(args, _STARPU_FUT_SCC_KEY);

	_starpu_scc_src_init_context(devid);

	args->status = STATUS_UNKNOWN;

	for (i = 0; i < config->topology.nmiccores[mp_nodeid]; i++)
	{
		struct _starpu_worker *worker = &config->workers[baseworkerid+i];
		snprintf(worker->name, sizeof(worker->name), "MIC %d core %u", mp_nodeid, i);
	}

	_STARPU_TRACE_WORKER_INIT_END;

	/* tell the main thread that this one is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&args->mutex);
	args->worker_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&args->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&args->mutex);

	_starpu_src_common_worker(args, baseworkerid, scc_mp_nodes[mp_nodeid]);

	_STARPU_TRACE_WORKER_DEINIT_START;

	_starpu_scc_src_deinit_context(args->devid);

	_STARPU_TRACE_WORKER_DEINIT_END(_STARPU_FUT_SCC_KEY);

	return NULL;
}
