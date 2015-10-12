/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012, 2013, 2014, 2015  CNRS
 * Copyright (C) 2011-2015  Université de Bordeaux
 * Copyright (C) 2014 INRIA
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
#include <starpu_mpi.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_comm.h>
#include <common/list.h>

struct _starpu_mpi_comm
{
	MPI_Comm comm;
	struct _starpu_mpi_envelope *envelope;
	MPI_Request request;
	int posted;
};
struct _starpu_mpi_comm_hashtable
{
	UT_hash_handle hh;
	MPI_Comm comm;
};
struct _starpu_mpi_comm_hashtable *_starpu_mpi_comms_cache;
struct _starpu_mpi_comm **_starpu_mpi_comms;
int _starpu_mpi_comm_nb;
int _starpu_mpi_comm_allocated;
int _starpu_mpi_comm_tested;

void _starpu_mpi_comm_init(MPI_Comm comm)
{
	_STARPU_MPI_DEBUG(10, "allocating for %d communicators\n", _starpu_mpi_comm_allocated);
	_starpu_mpi_comm_allocated=10;
	_starpu_mpi_comms = calloc(_starpu_mpi_comm_allocated, sizeof(struct _starpu_mpi_comm *));
	_starpu_mpi_comm_nb=0;
	_starpu_mpi_comm_tested=0;
	_starpu_mpi_comms_cache = NULL;

	_starpu_mpi_comm_register(comm);
}

void _starpu_mpi_comm_free()
{
	int i;
	for(i=0 ; i<_starpu_mpi_comm_nb ; i++)
	{
		struct _starpu_mpi_comm *_comm = _starpu_mpi_comms[i]; // get the ith _comm;
		free(_comm->envelope);
		free(_comm);
	}
	free(_starpu_mpi_comms);

	struct _starpu_mpi_comm_hashtable *entry, *tmp;
	HASH_ITER(hh, _starpu_mpi_comms_cache, entry, tmp)
	{
		HASH_DEL(_starpu_mpi_comms_cache, entry);
		free(entry);
	}
}

void _starpu_mpi_comm_register(MPI_Comm comm)
{
	struct _starpu_mpi_comm_hashtable *found;

	HASH_FIND(hh, _starpu_mpi_comms_cache, &comm, sizeof(MPI_Comm), found);
	if (found)
	{
		_STARPU_MPI_DEBUG(10, "comm %p (%p) already registered\n", comm, MPI_COMM_WORLD);
	}
	else
	{
		if (_starpu_mpi_comm_nb == _starpu_mpi_comm_allocated)
		{
			_starpu_mpi_comm_allocated *= 2;
			_STARPU_MPI_DEBUG(10, "reallocating for %d communicators\n", _starpu_mpi_comm_allocated);
			_starpu_mpi_comms = realloc(_starpu_mpi_comms, _starpu_mpi_comm_allocated * sizeof(struct _starpu_mpi_comm *));
		}
		_STARPU_MPI_DEBUG(10, "registering comm %p (%p) number %d\n", comm, MPI_COMM_WORLD, _starpu_mpi_comm_nb);
		struct _starpu_mpi_comm *_comm = calloc(1, sizeof(struct _starpu_mpi_comm));
		_comm->comm = comm;
		_comm->envelope = calloc(1,sizeof(struct _starpu_mpi_envelope));
		_comm->posted = 0;
		_starpu_mpi_comms[_starpu_mpi_comm_nb] = _comm;
		_starpu_mpi_comm_nb++;
		struct _starpu_mpi_comm_hashtable *entry = (struct _starpu_mpi_comm_hashtable *)malloc(sizeof(*entry));
		entry->comm = comm;
		HASH_ADD(hh, _starpu_mpi_comms_cache, comm, sizeof(entry->comm), entry);
	}
}

void _starpu_mpi_comm_post_recv()
{
	int i;
	for(i=0 ; i<_starpu_mpi_comm_nb ; i++)
	{
		struct _starpu_mpi_comm *_comm = _starpu_mpi_comms[i]; // get the ith _comm;
		if (_comm->posted == 0)
		{
			_STARPU_MPI_DEBUG(3, "Posting a receive to get a data envelop on comm %d %p\n", i, _comm->comm);
			_STARPU_MPI_COMM_FROM_DEBUG(sizeof(struct _starpu_mpi_envelope), MPI_BYTE, MPI_ANY_SOURCE, _STARPU_MPI_TAG_ENVELOPE, _STARPU_MPI_TAG_ENVELOPE, _comm->comm);
			MPI_Irecv(_comm->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, MPI_ANY_SOURCE, _STARPU_MPI_TAG_ENVELOPE, _comm->comm, &_comm->request);
			_comm->posted = 1;
		}
	}
}

int _starpu_mpi_comm_test_recv(MPI_Status *status, struct _starpu_mpi_envelope **envelope, MPI_Comm *comm)
{
	int i=_starpu_mpi_comm_tested;
	while (1)
	{
		int flag, res;

		struct _starpu_mpi_comm *_comm = _starpu_mpi_comms[i]; // get the ith _comm;

		if (_comm->posted)
		{
			/* test whether an envelope has arrived. */
#ifdef STARPU_SIMGRID
			MSG_process_sleep(0.000010);
#endif
			res = MPI_Test(&_comm->request, &flag, status);
			STARPU_ASSERT(res == MPI_SUCCESS);
			if (flag)
			{
				_comm->posted = 0;
				_starpu_mpi_comm_tested++;
				if (_starpu_mpi_comm_tested == _starpu_mpi_comm_nb)
					_starpu_mpi_comm_tested = 0;
				*envelope = _comm->envelope;
				*comm = _comm->comm;
				return 1;
			}
		}
		i++;
		if (i == _starpu_mpi_comm_nb) i=0;
		if (i == _starpu_mpi_comm_tested)
			// We have tested all the requests, none has completed
			return 0;
	}
	return 0;
}

void _starpu_mpi_comm_cancel_recv()
{
	int i;
	for(i=0 ; i<_starpu_mpi_comm_nb ; i++)
	{
		struct _starpu_mpi_comm *_comm = _starpu_mpi_comms[i]; // get the ith _comm;
		if (_comm->posted == 1)
		{
			MPI_Status status;
			MPI_Cancel(&_comm->request);
			MPI_Wait(&_comm->request, &status);
			_comm->posted = 0;
		}
	}
}
