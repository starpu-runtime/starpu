/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2014  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
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

#include <stdlib.h>
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_early_request.h>
#include <common/uthash.h>

/** stores application requests for which data have not been received yet */
static struct _starpu_mpi_req **_starpu_mpi_app_req_hashmap = NULL;
static int _starpu_mpi_app_req_hashmap_count = 0;

void _starpu_mpi_early_request_init(int world_size)
{
	int k;

	_starpu_mpi_app_req_hashmap = malloc(world_size * sizeof(struct _starpu_mpi_req *));
	for(k=0 ; k<world_size ; k++) _starpu_mpi_app_req_hashmap[k] = NULL;
}

void _starpu_mpi_early_request_free()
{
	free(_starpu_mpi_app_req_hashmap);
}

int _starpu_mpi_early_request_count()
{
	return _starpu_mpi_app_req_hashmap_count;
}

void _starpu_mpi_early_request_check_termination()
{
	STARPU_ASSERT_MSG(_starpu_mpi_early_request_count() == 0, "Number of receive requests left is not zero");
}

struct _starpu_mpi_req* _starpu_mpi_early_request_find(int mpi_tag, int source)
{
	struct _starpu_mpi_req* req;

	HASH_FIND_INT(_starpu_mpi_app_req_hashmap[source], &mpi_tag, req);

	return req;
}

void _starpu_mpi_early_request_add(struct _starpu_mpi_req *req)
{
	struct _starpu_mpi_req *test_req;

	test_req = _starpu_mpi_early_request_find(req->mpi_tag, req->srcdst);

	if (test_req == NULL)
	{
		HASH_ADD_INT(_starpu_mpi_app_req_hashmap[req->srcdst], mpi_tag, req);
		_starpu_mpi_app_req_hashmap_count ++;
		_STARPU_MPI_DEBUG(3, "Adding request %p with tag %d in the application request hashmap[%d]\n", req, req->mpi_tag, req->srcdst);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "[Error] request %p with tag %d already in the application request hashmap[%d]\n", req, req->mpi_tag, req->srcdst);
		int seq_const = starpu_data_get_sequential_consistency_flag(req->data_handle);
		if (seq_const &&  req->sequential_consistency)
		{
			STARPU_ASSERT_MSG(!test_req, "[Error] request %p with tag %d wanted to be added to the application request hashmap[%d], while another request %p with the same tag is already in it. \n Sequential consistency is activated : this is not supported by StarPU.", req, req->mpi_tag, req->srcdst, test_req);
		}
		else
		{
			STARPU_ASSERT_MSG(!test_req, "[Error] request %p with tag %d wanted to be added to the application request hashmap[%d], while another request %p with the same tag is already in it. \n Sequential consistency isn't activated for this handle : you should want to add dependencies between requests for which the sequential consistency is deactivated.", req, req->mpi_tag, req->srcdst, test_req);
		}
	}
}

void _starpu_mpi_early_request_delete(struct _starpu_mpi_req *req)
{
	struct _starpu_mpi_req *test_req;

	test_req = _starpu_mpi_early_request_find(req->mpi_tag, req->srcdst);

	if (test_req != NULL)
	{
		HASH_DEL(_starpu_mpi_app_req_hashmap[req->srcdst], req);
		_starpu_mpi_app_req_hashmap_count --;
		_STARPU_MPI_DEBUG(3, "Deleting application request %p with tag %d from the application request hashmap[%d]\n", req, req->mpi_tag, req->srcdst);
	}
	else
	{
		_STARPU_MPI_DEBUG(3, "[Warning] request %p with tag %d is NOT in the application request hashmap[%d]\n", req, req->mpi_tag, req->srcdst);
	}
}

