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

#include <sys/mman.h>
#include <fcntl.h>

#include <core/workers.h>

#include <drivers/mp_common/mp_common.h>
#include <drivers/scc/driver_scc_common.h>

#include <RCCE_lib.h>
#include <SCC_API.h>

static int rcce_initialized;

static int src_node_id;

static t_vcharp rckncm_map;
static t_vcharp shm_addr;


static void _starpu_scc_set_src_node_id()
{
	int node_id = starpu_get_env_number("STARPU_SCC_MASTER_NODE");

	if (node_id != -1)
	{
		if (node_id < RCCE_num_ues())
		{
			src_node_id = node_id;
			return;
		}
		else if (RCCE_ue() == 0)
		{
			/* Only node 0 print the error message. */
			fprintf(stderr, "The node you specify to be the master is "
					"greater than the total number of nodes.\n"
					"Taking node 0 (core %d) by default...\n", RC_COREID[0]);
		}
	}

	/* Node 0 by default. */
	src_node_id = 0;
}

/* Try to init the RCCE API.
 * return: 	1 on success
 * 			0 on failure
 */
int _starpu_scc_common_mp_init()
{
	int rckncm_fd;

	/* "/dev/rckncm" is to access shared memory on SCC. */
	if ((rckncm_fd = open("/dev/rckncm", O_RDWR | O_SYNC)) < 0)
	{
		/* It seems that we're not on a SCC system. */
		return (rcce_initialized = 0);
	}

	int page_size = getpagesize();
	unsigned int aligne_addr = (SHM_ADDR) & (~(page_size - 1));
	if ((rckncm_map = (t_vcharp)mmap(NULL, SHMSIZE, PROT_WRITE | PROT_READ, MAP_SHARED,
					rckncm_fd, aligne_addr)) == MAP_FAILED)
	{
		perror("mmap");
		close(rckncm_fd);
		return (rcce_initialized = 0);
	}

	int *argc = _starpu_get_argc();
	char ***argv = _starpu_get_argv();

	/* We can't initialize RCCE without argc and argv. */
	if (!argc || *argc <= 1 || !argv || (RCCE_init(argc, argv) != RCCE_SUCCESS))
	{
		close(rckncm_fd);
		munmap((void*)rckncm_map, SHMSIZE);
		return (rcce_initialized = 0);
	}

	unsigned int page_offset = (SHM_ADDR) - aligne_addr;
	shm_addr = rckncm_map + page_offset;

	RCCE_shmalloc_init(shm_addr, RCCE_SHM_SIZE_MAX);

	/* Which core of the SCC will be the master one? */
	_starpu_scc_set_src_node_id();

	close(rckncm_fd);

	return (rcce_initialized = 1);
}

void *_starpu_scc_common_get_shared_memory_addr()
{
	return (void*)shm_addr;
}

void _starpu_scc_common_unmap_shared_memory()
{
	munmap((void*)rckncm_map, SHMSIZE);
}

/* To know if the pointer "ptr" points into the shared memory map */
int _starpu_scc_common_is_in_shared_memory(void *ptr)
{
	return (void*)shm_addr <= ptr && ptr < (void*)shm_addr + SHMSIZE;
}

int _starpu_scc_common_is_mp_initialized()
{
	return rcce_initialized;
}

int _starpu_scc_common_get_src_node_id()
{
	return src_node_id;
}

int _starpu_scc_common_is_src_node()
{
	return RCCE_ue() == src_node_id;
}

void _starpu_scc_common_send(const struct _starpu_mp_node *node, void *msg, int len)
{
	int ret;

	/* There are potentially 48 threads running on the master core and RCCE_send write
	 * data in the MPB associated to this core. It's not thread safe, so we have to protect it.
	 * RCCE_acquire_lock uses a test&set register on SCC. */
	RCCE_acquire_lock(RCCE_ue());

	if ((ret = RCCE_send(msg, len, node->mp_connection.scc_nodeid)) != RCCE_SUCCESS)
	{
		RCCE_release_lock(RCCE_ue());
		STARPU_MP_COMMON_REPORT_ERROR(node, ret);
	}

	RCCE_release_lock(RCCE_ue());
}

void _starpu_scc_common_recv(const struct _starpu_mp_node *node, void *msg, int len)
{
	int ret;
	if ((ret = RCCE_recv(msg, len, node->mp_connection.scc_nodeid)) != RCCE_SUCCESS)
		STARPU_MP_COMMON_REPORT_ERROR(node, ret);
}

void _starpu_scc_common_report_rcce_error(const char *func, const char *file, const int line, const int err_no)
{
	char error_string[RCCE_MAX_ERROR_STRING];
	int error_string_length;

	RCCE_error_string(err_no, error_string, &error_string_length); 

	fprintf(stderr, "RCCE error in %s (%s:%d): %s\n", func, file, line, error_string); 
	STARPU_ABORT();
}
