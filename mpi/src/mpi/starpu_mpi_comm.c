/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2017       Guillaume Beauchamp
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
#include <mpi/starpu_mpi_comm.h>
#include <common/list.h>

#ifdef STARPU_USE_MPI_MPI

struct _starpu_mpi_comm
{
	MPI_Comm comm;
	struct _starpu_mpi_envelope *envelope;
	MPI_Request request;
	int posted;

#ifdef STARPU_SIMGRID
	MPI_Status status;
	starpu_pthread_queue_t queue;
	unsigned done;
#endif
};
struct _starpu_mpi_comm_hashtable
{
	UT_hash_handle hh;
	MPI_Comm comm;
};

/* Protect between comm addition from submitting tasks and MPI thread */
static starpu_pthread_rwlock_t _starpu_mpi_comms_mutex;

struct _starpu_mpi_comm_hashtable *_starpu_mpi_comms_cache;
struct _starpu_mpi_comm **_starpu_mpi_comms;
int _starpu_mpi_comm_nb;
int _starpu_mpi_comm_allocated;
int _starpu_mpi_comm_tested;

void _starpu_mpi_comm_init(MPI_Comm comm)
{
	_STARPU_MPI_DEBUG(10, "allocating for %d communicators\n", _starpu_mpi_comm_allocated);
	_starpu_mpi_comm_allocated=10;
	_STARPU_MPI_CALLOC(_starpu_mpi_comms, _starpu_mpi_comm_allocated, sizeof(struct _starpu_mpi_comm *));
	_starpu_mpi_comm_nb=0;
	_starpu_mpi_comm_tested=0;
	_starpu_mpi_comms_cache = NULL;
	STARPU_PTHREAD_RWLOCK_INIT(&_starpu_mpi_comms_mutex, NULL);

	_starpu_mpi_comm_register(comm);
}

void _starpu_mpi_comm_shutdown()
{
	int i;
	for(i=0 ; i<_starpu_mpi_comm_nb ; i++)
	{
		struct _starpu_mpi_comm *_comm = _starpu_mpi_comms[i]; // get the ith _comm;
		free(_comm->envelope);
#ifdef STARPU_SIMGRID
		starpu_pthread_queue_unregister(&_starpu_mpi_thread_wait, &_comm->queue);
		starpu_pthread_queue_destroy(&_comm->queue);
#endif
		free(_comm);
	}
	free(_starpu_mpi_comms);

	struct _starpu_mpi_comm_hashtable *entry=NULL, *tmp=NULL;
	HASH_ITER(hh, _starpu_mpi_comms_cache, entry, tmp)
	{
		HASH_DEL(_starpu_mpi_comms_cache, entry);
		free(entry);
	}

	STARPU_PTHREAD_RWLOCK_DESTROY(&_starpu_mpi_comms_mutex);
}

void _starpu_mpi_comm_register(MPI_Comm comm)
{
	struct _starpu_mpi_comm_hashtable *found;

	STARPU_PTHREAD_RWLOCK_RDLOCK(&_starpu_mpi_comms_mutex);
	HASH_FIND(hh, _starpu_mpi_comms_cache, &comm, sizeof(MPI_Comm), found);
	STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_mpi_comms_mutex);
	if (found)
	{
		_STARPU_MPI_DEBUG(10, "comm %ld (%ld) already registered\n", (long int)comm, (long int)MPI_COMM_WORLD);
		return;
	}

	STARPU_PTHREAD_RWLOCK_WRLOCK(&_starpu_mpi_comms_mutex);
	HASH_FIND(hh, _starpu_mpi_comms_cache, &comm, sizeof(MPI_Comm), found);
	if (found)
	{
		_STARPU_MPI_DEBUG(10, "comm %ld (%ld) already registered in between\n", (long int)comm, (long int)MPI_COMM_WORLD);
	}
	else
	{
		if (_starpu_mpi_comm_nb == _starpu_mpi_comm_allocated)
		{
			_starpu_mpi_comm_allocated *= 2;
			_STARPU_MPI_DEBUG(10, "reallocating for %d communicators\n", _starpu_mpi_comm_allocated);
			_STARPU_MPI_REALLOC(_starpu_mpi_comms, _starpu_mpi_comm_allocated * sizeof(struct _starpu_mpi_comm *));
		}
		_STARPU_MPI_DEBUG(10, "registering comm %ld (%ld) number %d\n", (long int)comm, (long int)MPI_COMM_WORLD, _starpu_mpi_comm_nb);
		struct _starpu_mpi_comm *_comm;
		_STARPU_MPI_CALLOC(_comm, 1, sizeof(struct _starpu_mpi_comm));
		_comm->comm = comm;
		_STARPU_MPI_CALLOC(_comm->envelope, 1,sizeof(struct _starpu_mpi_envelope));
		_comm->posted = 0;
		_starpu_mpi_comms[_starpu_mpi_comm_nb] = _comm;
		_starpu_mpi_comm_nb++;
		struct _starpu_mpi_comm_hashtable *entry;
		_STARPU_MPI_MALLOC(entry, sizeof(*entry));
		entry->comm = comm;
		HASH_ADD(hh, _starpu_mpi_comms_cache, comm, sizeof(entry->comm), entry);

#ifdef STARPU_SIMGRID
		starpu_pthread_queue_init(&_comm->queue);
		starpu_pthread_queue_register(&_starpu_mpi_thread_wait, &_comm->queue);
		_comm->done = 0;
#endif
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_mpi_comms_mutex);
}

void _starpu_mpi_comm_post_recv()
{
	int i;

	STARPU_PTHREAD_RWLOCK_RDLOCK(&_starpu_mpi_comms_mutex);
	for(i=0 ; i<_starpu_mpi_comm_nb ; i++)
	{
		struct _starpu_mpi_comm *_comm = _starpu_mpi_comms[i]; // get the ith _comm;
		if (_comm->posted == 0)
		{
			_STARPU_MPI_DEBUG(3, "Posting a receive to get a data envelop on comm %d %ld\n", i, (long int)_comm->comm);
			_STARPU_MPI_COMM_FROM_DEBUG(_comm->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, MPI_ANY_SOURCE, _STARPU_MPI_TAG_ENVELOPE, (int64_t)_STARPU_MPI_TAG_ENVELOPE, _comm->comm);
			MPI_Irecv(_comm->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, MPI_ANY_SOURCE, _STARPU_MPI_TAG_ENVELOPE, _comm->comm, &_comm->request);
#ifdef STARPU_SIMGRID
			_starpu_mpi_simgrid_wait_req(&_comm->request, &_comm->status, &_comm->queue, &_comm->done);
#endif
			_comm->posted = 1;
		}
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_mpi_comms_mutex);
}

int _starpu_mpi_comm_test_recv(MPI_Status *status, struct _starpu_mpi_envelope **envelope, MPI_Comm *comm)
{
	int i=_starpu_mpi_comm_tested;

	STARPU_PTHREAD_RWLOCK_RDLOCK(&_starpu_mpi_comms_mutex);
	while (1)
	{
		struct _starpu_mpi_comm *_comm = _starpu_mpi_comms[i]; // get the ith _comm;

		if (_comm->posted)
		{
			int flag, res;
			/* test whether an envelope has arrived. */
#ifdef STARPU_SIMGRID
			res = _starpu_mpi_simgrid_mpi_test(&_comm->done, &flag);
			memcpy(status, &_comm->status, sizeof(*status));
#else
			res = MPI_Test(&_comm->request, &flag, status);
#endif
			STARPU_ASSERT(res == MPI_SUCCESS);
			if (flag)
			{
				_comm->posted = 0;
				_starpu_mpi_comm_tested++;
				if (_starpu_mpi_comm_tested == _starpu_mpi_comm_nb)
					_starpu_mpi_comm_tested = 0;
				*envelope = _comm->envelope;
				*comm = _comm->comm;
				STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_mpi_comms_mutex);
				return 1;
			}
		}
		i++;
		if (i == _starpu_mpi_comm_nb)
		{
			i=0;
		}
		if (i == _starpu_mpi_comm_tested)
		{
			// We have tested all the requests, none has completed
			STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_mpi_comms_mutex);
			return 0;
		}
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_mpi_comms_mutex);
	return 0;
}

void _starpu_mpi_comm_cancel_recv()
{
	int i;

	STARPU_PTHREAD_RWLOCK_RDLOCK(&_starpu_mpi_comms_mutex);
	for(i=0 ; i<_starpu_mpi_comm_nb ; i++)
	{
		struct _starpu_mpi_comm *_comm = _starpu_mpi_comms[i]; // get the ith _comm;
		if (_comm->posted == 1)
		{
			MPI_Cancel(&_comm->request);
#ifndef STARPU_SIMGRID
			{
				MPI_Status status;
				MPI_Wait(&_comm->request, &status);
			}
#endif
			_comm->posted = 0;
		}
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_mpi_comms_mutex);
}

#endif /* STARPU_USE_MPI_MPI */
