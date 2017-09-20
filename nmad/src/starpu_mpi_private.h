/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012-2015, 2017  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015  Centre National de la Recherche Scientifique
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
#include <starpu_mpi.h>
#include <starpu_mpi_fxt.h>
#include <common/list.h>
#include <common/prio_list.h>
#include <core/simgrid.h>
#include <pioman.h>
#include <nm_sendrecv_interface.h>
#include <nm_session_interface.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef STARPU_SIMGRID
starpu_pthread_wait_t wait;
starpu_pthread_queue_t dontsleep;

struct _starpu_simgrid_mpi_req
{
	MPI_Request *request;
	MPI_Status *status;
	starpu_pthread_queue_t *queue;
	unsigned *done;
};

int _starpu_mpi_simgrid_mpi_test(unsigned *done, int *flag);
void _starpu_mpi_simgrid_wait_req(MPI_Request *request, 	MPI_Status *status, starpu_pthread_queue_t *queue, unsigned *done);
#endif

extern int _starpu_debug_rank;
char *_starpu_mpi_get_mpi_error_code(int code);
extern int _starpu_mpi_comm_debug;

#ifdef STARPU_VERBOSE
extern int _starpu_debug_level_min;
extern int _starpu_debug_level_max;
void _starpu_mpi_set_debug_level_min(int level);
void _starpu_mpi_set_debug_level_max(int level);
#endif
extern int _starpu_mpi_fake_world_size;
extern int _starpu_mpi_fake_world_rank;

#ifdef STARPU_NO_ASSERT
#  define STARPU_MPI_ASSERT_MSG(x, msg, ...)	do { if (0) { (void) (x); }} while(0)
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

#define _STARPU_MPI_MALLOC(ptr, size) do { ptr = malloc(size); STARPU_MPI_ASSERT_MSG(ptr != NULL, "Cannot allocate %ld bytes\n", (long) size); } while (0)
#define _STARPU_MPI_CALLOC(ptr, nmemb, size) do { ptr = calloc(nmemb, size); STARPU_MPI_ASSERT_MSG(ptr != NULL, "Cannot allocate %ld bytes\n", (long) (nmemb*size)); } while (0)
#define _STARPU_MPI_REALLOC(ptr, size) do { void *_new_ptr = realloc(ptr, size); STARPU_MPI_ASSERT_MSG(_new_ptr != NULL, "Cannot reallocate %ld bytes\n", (long) size); ptr = _new_ptr; } while (0)

#ifdef STARPU_VERBOSE
#  define _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, node, tag, utag, comm, way) \
	do								\
	{							\
	     	if (_starpu_mpi_comm_debug)			\
		{					\
     			int __size;			\
			char _comm_name[128];		\
			int _comm_name_len;		\
			int _rank;			    \
			starpu_mpi_comm_rank(comm, &_rank); \
			MPI_Type_size(datatype, &__size);		\
			MPI_Comm_get_name(comm, _comm_name, &_comm_name_len); \
			fprintf(stderr, "[%d][starpu_mpi] :%d:%s:%d:%d:%d:%s:%p:%ld:%d:%s:%d\n", _rank, _rank, way, node, tag, utag, _comm_name, ptr, count, __size, __starpu_func__ , __LINE__); \
			fflush(stderr);					\
		}							\
	} while(0);
#  define _STARPU_MPI_COMM_TO_DEBUG(ptr, count, datatype, dest, tag, utag, comm) 	    _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, dest, tag, utag, comm, "-->")
#  define _STARPU_MPI_COMM_FROM_DEBUG(ptr, count, datatype, source, tag, utag, comm)  _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, source, tag, utag, comm, "<--")
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
#  define _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, node, tag, utag, comm, way)  do { } while(0)
#  define _STARPU_MPI_COMM_TO_DEBUG(ptr, count, datatype, dest, tag, utag, comm)     do { } while(0)
#  define _STARPU_MPI_COMM_FROM_DEBUG(ptr, count, datatype, source, tag, utag, comm) do { } while(0)
#  define _STARPU_MPI_DEBUG(level, fmt, ...)		do { } while(0)
#endif

#define _STARPU_MPI_DISP(fmt, ...) do { if (!_starpu_silent) { \
	       				     if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
                                             fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] " fmt , (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__ , __LINE__ ,## __VA_ARGS__); \
                                             fflush(stderr); }} while(0);
#define _STARPU_MPI_MSG(fmt, ...) do { if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
                                             fprintf(stderr, "[%d][starpu_mpi][%s:%d] " fmt , _starpu_debug_rank, __starpu_func__ , __LINE__ ,## __VA_ARGS__); \
                                             fflush(stderr); } while(0);

#ifdef STARPU_VERBOSE
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

struct _starpu_mpi_node_tag
{
	MPI_Comm comm;
	int rank;
	int data_tag;
};

struct _starpu_mpi_data
{
	int magic;
	struct _starpu_mpi_node_tag node_tag;
	int *cache_sent;
	int cache_received;
};

LIST_TYPE(_starpu_mpi_req,
	/* description of the data at StarPU level */
	starpu_data_handle_t data_handle;

	int prio;

	/* description of the data to be sent/received */
	MPI_Datatype datatype;
	char *datatype_name;
	void *ptr;
	starpu_ssize_t count;
	int registered_datatype;

	/* who are we talking to ? */
	struct _starpu_mpi_node_tag node_tag;
	nm_gate_t gate;
	nm_session_t session;

	void (*func)(struct _starpu_mpi_req *);

	MPI_Status *status;
	nm_sr_request_t request;
	int *flag;
	unsigned sync;

	int ret;
	piom_cond_t req_cond;

	enum _starpu_mpi_request_type request_type; /* 0 send, 1 recv */

	unsigned submitted;
	unsigned completed;
	unsigned posted;

	/* in the case of detached requests */
	int detached;
	void *callback_arg;
	void (*callback)(void *);

        /* in the case of user-defined datatypes, we need to send the size of the data */
	nm_sr_request_t size_req;

	int sequential_consistency;

	long pre_sync_jobid;
	long post_sync_jobid;

	int waited;

#ifdef STARPU_SIMGRID
        MPI_Status status_store;
	starpu_pthread_queue_t queue;
	unsigned done;
#endif
);
PRIO_LIST_TYPE(_starpu_mpi_req, prio)

struct _starpu_mpi_argc_argv
{
	int initialize_mpi;
	int *argc;
	char ***argv;
	MPI_Comm comm;
	int fargc;	// Fortran argc
	char **fargv;	// Fortran argv
	int rank;
	int world_size;
};

void _starpu_mpi_progress_shutdown(int *value);
int _starpu_mpi_progress_init(struct _starpu_mpi_argc_argv *argc_argv);
#ifdef STARPU_SIMGRID
void _starpu_mpi_wait_for_initialization();
#endif

#define _starpu_mpi_req_status(PUBLIC_REQ,STATUS) do {			\
	STATUS->MPI_SOURCE=PUBLIC_REQ->node_tag.rank; /**< field name mandatory by spec */ \
	STATUS->MPI_TAG=PUBLIC_REQ->node_tag.data_tag;    /**< field name mandatory by spec */ \
	STATUS->MPI_ERROR=PUBLIC_REQ->ret;  /**< field name mandatory by spec */ \
	STATUS->size=PUBLIC_REQ->count;       /**< size of data received */ \
	STATUS->cancelled=0;  /**< whether request was cancelled */	\
} while(0)

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_PRIVATE_H__
