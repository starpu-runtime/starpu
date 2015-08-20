/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012-2015  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015  CNRS
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

#ifndef __STARPU_MPI_PRIVATE_H__
#define __STARPU_MPI_PRIVATE_H__

#include <starpu.h>
#include <common/config.h>
#include <common/uthash.h>
#include "starpu_mpi.h"
#include "starpu_mpi_fxt.h"
#include <common/list.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int _starpu_debug_rank;
char *_starpu_mpi_get_mpi_code(int code);
extern int _starpu_mpi_comm;

#ifdef STARPU_VERBOSE
extern int _starpu_debug_level_min;
extern int _starpu_debug_level_max;
void _starpu_mpi_set_debug_level_min(int level);
void _starpu_mpi_set_debug_level_max(int level);
#endif

#ifdef STARPU_NO_ASSERT
#  define STARPU_MPI_ASSERT_MSG(x, msg, ...)	do { } while(0)
#else
#  if defined(__CUDACC__) && defined(STARPU_HAVE_WINDOWS)
int _starpu_debug_rank;
#    define STARPU_MPI_ASSERT_MSG(x, msg, ...)									\
	do													\
	{ 													\
		if (STARPU_UNLIKELY(!(x))) 									\
		{												\
			if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
			fprintf(stderr, "\n[%d][starpu_mpi][%s][assert failure] " msg "\n\n", _starpu_debug_rank, __starpu_func__, ## __VA_ARGS__); *(int*)NULL = 0; \
		} \
	} while(0)
#  else
#    define STARPU_MPI_ASSERT_MSG(x, msg, ...)	\
	do \
	{ \
		if (STARPU_UNLIKELY(!(x))) \
		{ \
			if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
			fprintf(stderr, "\n[%d][starpu_mpi][%s][assert failure] " msg "\n\n", _starpu_debug_rank, __starpu_func__, ## __VA_ARGS__); \
		} \
		assert(x); \
	} while(0)

#  endif
#endif

#ifdef STARPU_VERBOSE
#  define _STARPU_MPI_COMM_DEBUG(count, datatype, node, tag, utag, comm, way) \
	do \
	{ \
	     	if (_starpu_mpi_comm)	\
	     	{ \
     			int __size; \
			if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
			MPI_Type_size(datatype, &__size); \
			fprintf(stderr, "[%d][starpu_mpi] %s %d:%d(%d):%p %12s %ld     [%s:%d]\n", _starpu_debug_rank, way, node, tag, utag, comm, " ", count*__size, __starpu_func__ , __LINE__); \
			fflush(stderr); \
		} \
	} while(0);
#  define _STARPU_MPI_COMM_TO_DEBUG(count, datatype, dest, tag, utag, comm) 		_STARPU_MPI_COMM_DEBUG(count, datatype, dest, tag, utag, comm, "-->")
#  define _STARPU_MPI_COMM_FROM_DEBUG(count, datatype, source, tag, utag, comm) 	_STARPU_MPI_COMM_DEBUG(count, datatype, source, tag, utag, comm, "<--")
#  define _STARPU_MPI_DEBUG(level, fmt, ...) \
	do \
	{								\
		if (!_starpu_silent && _starpu_debug_level_min <= level && level <= _starpu_debug_level_max)	\
		{							\
			if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
			fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] " fmt , (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__ , __LINE__,## __VA_ARGS__); \
			fflush(stderr); \
		}			\
	} while(0);
#else
#  define _STARPU_MPI_COMM_DEBUG(count, datatype, node, tag, utag, comm, way)		do { } while(0)
#  define _STARPU_MPI_COMM_TO_DEBUG(count, datatype, dest, tag, comm, utag)		do { } while(0)
#  define _STARPU_MPI_COMM_FROM_DEBUG(count, datatype, source, tag, comm, utag)	do { } while(0)
#  define _STARPU_MPI_DEBUG(level, fmt, ...)		do { } while(0)
#endif

#define _STARPU_MPI_DISP(fmt, ...) do { if (!_starpu_silent) { \
	       				     if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
                                             fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] " fmt , (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__ , __LINE__ ,## __VA_ARGS__); \
                                             fflush(stderr); }} while(0);
#define _STARPU_MPI_MSG(fmt, ...) do { if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
                                             fprintf(stderr, "[%d][starpu_mpi][%s:%d] " fmt , _starpu_debug_rank, __starpu_func__ , __LINE__ ,## __VA_ARGS__); \
                                             fflush(stderr); } while(0);

#ifdef STARPU_VERBOSE0
#  define _STARPU_MPI_LOG_IN()             do { if (!_starpu_silent) { \
                                               if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank);                        \
                                               fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] -->\n", (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__ , __LINE__); \
                                               fflush(stderr); }} while(0)
#  define _STARPU_MPI_LOG_OUT()            do { if (!_starpu_silent) { \
                                               if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank);                        \
                                               fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] <--\n", (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__, __LINE__ ); \
                                               fflush(stderr); }} while(0)
#else
#  define _STARPU_MPI_LOG_IN()
#  define _STARPU_MPI_LOG_OUT()
#endif

extern int _starpu_mpi_tag;
#define _STARPU_MPI_TAG_ENVELOPE  _starpu_mpi_tag
#define _STARPU_MPI_TAG_DATA      _starpu_mpi_tag+1
#define _STARPU_MPI_TAG_SYNC_DATA _starpu_mpi_tag+2

enum _starpu_mpi_request_type
{
	SEND_REQ=0,
	RECV_REQ=1,
	WAIT_REQ=2,
	TEST_REQ=3,
	BARRIER_REQ=4,
	PROBE_REQ=5,
	UNKNOWN_REQ=6,
};

#define _STARPU_MPI_ENVELOPE_DATA       0
#define _STARPU_MPI_ENVELOPE_SYNC_READY 1

struct _starpu_mpi_envelope
{
	int mode;
	starpu_ssize_t size;
	int data_tag;
	unsigned sync;
};

struct _starpu_mpi_req;

struct _starpu_mpi_node_tag
{
	MPI_Comm comm;
	int rank;
	int data_tag;
};

LIST_TYPE(_starpu_mpi_req,
	/* description of the data at StarPU level */
	starpu_data_handle_t data_handle;

	/* description of the data to be sent/received */
	MPI_Datatype datatype;
	void *ptr;
	starpu_ssize_t count;
	int registered_datatype;

	/* who are we talking to ? */
	struct _starpu_mpi_node_tag node_tag;

	void (*func)(struct _starpu_mpi_req *);

	MPI_Status *status;
	MPI_Request data_request;
	int *flag;
	unsigned sync;

	int ret;
	starpu_pthread_mutex_t req_mutex;
	starpu_pthread_cond_t req_cond;

	starpu_pthread_mutex_t posted_mutex;
	starpu_pthread_cond_t posted_cond;

	enum _starpu_mpi_request_type request_type; /* 0 send, 1 recv */

	unsigned submitted;
	unsigned completed;
	unsigned posted;

	/* In the case of a Wait/Test request, we are going to post a request
	 * to test the completion of another request */
	struct _starpu_mpi_req *other_request;

	/* in the case of detached requests */
	int detached;
	void *callback_arg;
	void (*callback)(void *);

        /* in the case of user-defined datatypes, we need to send the size of the data */
	MPI_Request size_req;

        struct _starpu_mpi_envelope* envelope;

	int is_internal_req;
	struct _starpu_mpi_req *internal_req;

	int sequential_consistency;

     	UT_hash_handle hh;
);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_PRIVATE_H__
